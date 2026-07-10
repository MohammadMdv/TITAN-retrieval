"""Download a patient-disjoint, budget-constrained subset of the BRACS WSI set.

BRACS: 547 breast-carcinoma .svs WSIs from ~187 patients, 7 lesion classes
(N, PB, UDH, FEA, ADH, DCIS, IC), CC0. There is no public mirror and the portal does not
publish file sizes, so this script MEASURES every candidate over FTP before downloading.

Access: the BRACS FTP server is ANONYMOUS-ONLY. Registration on the website gates the portal
page that tells you the FTP host, not the FTP server itself -- so only BRACS_FTP_HOST is
required. (Supplying BRACS_FTP_USER/PASS is harmless: they are tried first, then anonymous.)

Why subset by PATIENT and never by slide
----------------------------------------
BRACS averages ~2.9 slides per patient. Selecting individual slides would place slides from
one patient on both sides of a train/test boundary -- precisely the leakage this project
exists to avoid. Every selection here is all-or-nothing at the patient level.

What this script does NOT assume
--------------------------------
BRACS's official 395:65:87 split is documented as slide-level; whether it is patient-disjoint
is not stated anywhere. This script CHECKS it against the metadata and reports the answer.
  - if disjoint  -> the subset preserves the official split assignment (so results stay
                    comparable to published BRACS numbers), budgeting each split separately.
  - if NOT       -> it says so loudly, pools all patients, and leaves split construction to a
                    later step. Do not silently reuse a leaking split.

Usage
-----
    # 1. put the FTP host in .env (see .env.example):  BRACS_FTP_HOST=...
    # 2. inspect the plan without downloading anything (fast, recommended first):
    python validation/download_bracs.py --dry-run
    # 3. run the real download (long; resumable -- safe to Ctrl-C and re-run):
    python validation/download_bracs.py

Resumable: a slide already on disk with the exact remote byte size is skipped; a partial file
is resumed with FTP REST. Interrupting and re-running never re-downloads completed slides.

Env: DISK_BUDGET_GB (default 150), MIN_PER_CLASS (default 5), BRACS_ROOT, SEED.
"""
import argparse
import ftplib
import os
import random
import ssl
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from common import TITAN_ROOT, load_env_file

BRACS_ROOT = Path(os.environ.get("BRACS_ROOT", TITAN_ROOT / "data" / "BRACS"))
DISK_BUDGET_GB = float(os.environ.get("DISK_BUDGET_GB", 150))
MIN_PER_CLASS = int(os.environ.get("MIN_PER_CLASS", 5))
SEED = int(os.environ.get("SEED", 0))

GB = 1024 ** 3
WSI_DIR_HINTS = ("whole slide image", "wsi")
CLASSES = ["N", "PB", "UDH", "FEA", "ADH", "DCIS", "IC"]


# --------------------------------------------------------------------------- FTP


class ReusedSslFTP_TLS(ftplib.FTP_TLS):
    """FTP_TLS that reuses the control channel's TLS session on the data channel.

    vsFTPd with `require_ssl_reuse=YES` (the BRACS server's setting) rejects data transfers
    that negotiate a fresh TLS session: "522 SSL connection failed: session reuse required".
    Stdlib FTP_TLS does not reuse, so plain FTPS logs in fine and then dies on the first LIST.
    """

    def ntransfercmd(self, cmd, rest=None):
        conn, size = ftplib.FTP.ntransfercmd(self, cmd, rest)
        if self._prot_p:
            conn = self.context.wrap_socket(conn, server_hostname=self.host,
                                            session=self.sock.session)
        return conn, size


def _try_login(host, user, pw, use_tls):
    """One connect+login attempt, VALIDATED by an actual data-channel round-trip.

    Logging in successfully is not enough: TLS data-channel problems only surface on the first
    transfer. So we issue a NLST here and let a failure fall through to the next candidate.
    """
    ftp = ReusedSslFTP_TLS(host, timeout=60) if use_tls else ftplib.FTP(host, timeout=60)
    try:
        ftp.login(user, pw)
        if use_tls:
            ftp.prot_p()
        ftp.set_pasv(True)
        ftp.voidcmd("TYPE I")  # binary mode; SIZE is only valid here
        ftp.nlst()             # prove the data channel actually works
        return ftp
    except Exception:
        try:
            ftp.close()
        except Exception:
            pass
        raise


def connect():
    """Log in to the BRACS FTP server.

    The BRACS server is ANONYMOUS-ONLY ("530 This FTP server is anonymous only"): registration
    gates the web portal, not FTP. So credentials are optional -- if BRACS_FTP_USER/PASS are
    set we try them first (in case the server ever changes), then fall back to anonymous.
    """
    host = os.environ.get("BRACS_FTP_HOST")
    if not host:
        sys.exit("[bracs] Missing BRACS_FTP_HOST. Set it in your .env (see .env.example).\n"
                 "        The BRACS FTP server allows anonymous access; no username needed.")

    user = os.environ.get("BRACS_FTP_USER")
    pw = os.environ.get("BRACS_FTP_PASS")
    attempts = []
    if user and pw:
        attempts.append((user, pw, "supplied credentials"))
    attempts.append(("anonymous", "anonymous@", "anonymous"))

    last = None
    for u, p, how in attempts:
        for use_tls in (True, False):
            try:
                ftp = _try_login(host, u, p, use_tls)
                print(f"[bracs] connected to {host} over "
                      f"{'FTPS' if use_tls else 'plain FTP'} ({how})")
                return ftp
            except (ftplib.error_perm, ftplib.error_temp, ftplib.error_proto,
                    ssl.SSLError, EOFError, OSError) as e:
                last = e
                if isinstance(e, ftplib.error_perm) and "anonymous only" in str(e).lower():
                    break  # credentials will never work here; skip straight to anonymous
    sys.exit(f"[bracs] could not log in to {host}: {last}")


def _parse_list_line(line):
    """Parse one Unix `ls -l` style LIST line -> (name, size_or_None). None size => directory.

    vsFTPd emits: `-rw-r--r--  1 0 0  1684552007 Jan 01 12:00 BRACS_300.svs`
    Names may contain spaces, so split at most 8 times and keep the remainder as the name.
    """
    parts = line.split(maxsplit=8)
    if len(parts) < 9:
        return None, None
    name = parts[8]
    if name in (".", ".."):
        return None, None
    if line[0] == "d":
        return name, None
    if line[0] == "l":  # symlink: "name -> target"
        name = name.split(" -> ")[0]
        return name, None
    try:
        return name, int(parts[4])
    except ValueError:
        return None, None


def listdir(ftp, path=""):
    """Return [(full_path, size_or_None)] for one directory. size None => it is a directory.

    The BRACS server is vsFTPd without MLSD ("500 Unknown command"). Rather than NLST plus a
    SIZE probe per entry (one round-trip each -- ~1.6 files/s over this link), we issue one
    LIST per directory, which already carries type and size for every entry. NLST+SIZE stays
    as a last-resort fallback for servers with unparseable LIST output.
    """
    try:
        out = []
        for name, facts in ftp.mlsd(path or "."):
            if name in (".", ".."):
                continue
            full = f"{path.rstrip('/')}/{name}" if path else name
            if facts.get("type") == "dir":
                out.append((full, None))
            elif facts.get("type") == "file":
                out.append((full, int(facts["size"]) if "size" in facts else ftp.size(full)))
        return out
    except (ftplib.error_perm, ftplib.error_proto, ftplib.error_temp):
        pass  # no MLSD -> LIST

    lines = []
    try:
        ftp.retrlines(f"LIST {path}" if path else "LIST", lines.append)
        out = []
        for line in lines:
            name, size = _parse_list_line(line)
            if name is None:
                continue
            out.append((f"{path.rstrip('/')}/{name}" if path else name, size))
        if out or not lines:
            return out
    except (ftplib.error_perm, ftplib.error_proto, ftplib.error_temp):
        pass  # unparseable LIST -> NLST + SIZE probe

    out = []
    for full in ftp.nlst(path):
        if full.rstrip("/").split("/")[-1] in (".", ".."):
            continue
        try:
            size = ftp.size(full)
        except (ftplib.error_perm, ftplib.error_temp):
            size = None
        out.append((full, size))
    return out


def walk(ftp, path):
    """Yield (remote_path, size_bytes) for every file beneath `path`."""
    for full, size in listdir(ftp, path):
        if size is None:
            yield from walk(ftp, full)
        else:
            yield full, size


def find_root(ftp):
    """Locate the WSI directory at the server root (BRACS_WSI).

    Must not match BRACS_WSI_Annotations (contains 'wsi') or BRACS_RoI.
    """
    if os.environ.get("BRACS_WSI_DIR"):
        return os.environ["BRACS_WSI_DIR"]
    cands = [p for p, size in listdir(ftp, "")
             if size is None
             and any(h in p.lower() for h in WSI_DIR_HINTS)
             and "annot" not in p.lower() and "roi" not in p.lower()]
    return min(cands, key=len) if cands else None


def find_metadata(ftp):
    """The BRACS.xlsx summary lives at the SERVER ROOT, not inside BRACS_WSI."""
    return [p for p, size in listdir(ftp, "")
            if size is not None and p.lower().endswith((".xlsx", ".xls"))]


# --------------------------------------------------------------- metadata parsing


def _pick_column(cols, *keywords):
    for c in cols:
        lc = str(c).strip().lower()
        if any(k in lc for k in keywords):
            return c
    return None


def parse_metadata(xlsx_path):
    """BRACS ships an .xlsx with, per WSI: label, reference set, patient ID, #ROIs.
    Column names are not documented, so match them case-insensitively by keyword."""
    df = pd.read_excel(xlsx_path)
    cols = list(df.columns)
    c_slide = _pick_column(cols, "wsi", "slide", "filename", "image")
    c_pat = _pick_column(cols, "patient", "case")
    c_lab = _pick_column(cols, "label", "type", "class", "diagnos")
    c_set = _pick_column(cols, "set", "split", "reference")
    missing = [n for n, c in [("slide", c_slide), ("patient", c_pat), ("label", c_lab)] if c is None]
    if missing:
        sys.exit(f"[bracs] could not find column(s) {missing} in {xlsx_path.name}.\n"
                 f"        Columns present: {cols}\n"
                 f"        Edit _pick_column() keywords to match.")
    print(f"[bracs] metadata columns -> slide={c_slide!r} patient={c_pat!r} "
          f"label={c_lab!r} split={c_set!r}")

    out = pd.DataFrame({
        "slide_id": df[c_slide].astype(str).str.strip().str.replace(r"\.svs$", "", regex=True),
        "patient_id": df[c_pat].astype(str).str.strip(),
        "label": df[c_lab].astype(str).str.strip(),
        "official_split": (df[c_set].astype(str).str.strip().str.lower()
                           if c_set else "unknown"),
    })
    return out.dropna(subset=["slide_id", "patient_id"]).drop_duplicates("slide_id")


def check_patient_disjoint(meta):
    """Is the official split patient-disjoint? Returns (bool, overlap_report)."""
    if (meta["official_split"] == "unknown").all():
        return False, "no split column found in metadata"
    groups = {s: set(g["patient_id"]) for s, g in meta.groupby("official_split")}
    overlaps = {}
    names = sorted(groups)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = groups[a] & groups[b]
            if shared:
                overlaps[f"{a}&{b}"] = sorted(shared)
    if overlaps:
        detail = "; ".join(f"{k}: {sorted(v)}" for k, v in overlaps.items())
        return False, detail
    return True, "train/val/test patient sets are pairwise disjoint"


def repair_splits(meta):
    """Make the official split patient-disjoint by assigning each straddling patient wholly
    to one split: the one where it already holds the most slides (ties -> the larger split).

    As of the current BRACS release exactly one patient straddles (id 67: 3 training slides,
    2 validation) and `testing` is already disjoint from both -- so the leak only ever
    contaminated hyperparameter selection, never test evaluation. Repairing costs a couple of
    validation slides and buys a split that is safe to select on. Returns (meta, moves).
    """
    meta = meta.copy()
    order = ["training", "testing", "validation"]  # tie-break preference: bigger split wins
    rank = {s: i for i, s in enumerate(order)}
    moves = []
    for pid, g in meta.groupby("patient_id"):
        splits = set(g["official_split"])
        if len(splits) <= 1:
            continue
        counts = g["official_split"].value_counts()
        keep = sorted(counts.index, key=lambda s: (-counts[s], rank.get(s, 99)))[0]
        for s in splits - {keep}:
            moves.append((pid, s, keep, int(counts[s])))
        meta.loc[meta["patient_id"] == pid, "official_split"] = keep
    return meta, moves


# ------------------------------------------------------------------- selection


def select_patients(meta, sizes, budget_bytes, min_per_class, rng):
    """Greedy, all-or-nothing per patient.

    Pass 1 guarantees class coverage (rarest class first, cheapest qualifying patient first)
    so tiny classes like FEA/ADH are not starved by the budget.
    Pass 2 spends what is left on the patients giving the most slides per byte.
    """
    by_pat = defaultdict(list)
    for r in meta.itertuples():
        if r.slide_id in sizes:
            by_pat[r.patient_id].append(r)
    if not by_pat:
        return [], 0

    cost = {p: sum(sizes[r.slide_id] for r in rows) for p, rows in by_pat.items()}
    labels_of = {p: [r.label for r in rows] for p, rows in by_pat.items()}

    chosen, spent = set(), 0
    class_counts = defaultdict(int)

    global_counts = defaultdict(int)
    for rows in by_pat.values():
        for r in rows:
            global_counts[r.label] += 1

    for cls in sorted(global_counts, key=lambda c: global_counts[c]):  # rarest first
        cands = [p for p in by_pat if p not in chosen and cls in labels_of[p]]
        cands.sort(key=lambda p: cost[p])                              # cheapest first
        for p in cands:
            if class_counts[cls] >= min_per_class:
                break
            if spent + cost[p] > budget_bytes:
                continue
            chosen.add(p)
            spent += cost[p]
            for lab in labels_of[p]:
                class_counts[lab] += 1

    remaining = [p for p in by_pat if p not in chosen]
    rng.shuffle(remaining)  # break ties among equal density deterministically
    remaining.sort(key=lambda p: len(by_pat[p]) / max(cost[p], 1), reverse=True)
    for p in remaining:
        if spent + cost[p] <= budget_bytes:
            chosen.add(p)
            spent += cost[p]

    return sorted(chosen), spent


# -------------------------------------------------------------------- download


def download_file(ftp, remote, local, size, overall):
    local.parent.mkdir(parents=True, exist_ok=True)
    have = local.stat().st_size if local.exists() else 0
    if have == size:
        overall.update(size)
        return "skip"
    if have > size:
        local.unlink()
        have = 0

    mode = "ab" if have else "wb"
    with open(local, mode) as fh, tqdm(
            total=size, initial=have, unit="B", unit_scale=True, unit_divisor=1024,
            desc=f"  {local.name[:28]:<28}", leave=False) as bar:
        def cb(chunk):
            fh.write(chunk)
            bar.update(len(chunk))
            overall.update(len(chunk))
        ftp.retrbinary(f"RETR {remote}", cb, blocksize=1 << 20, rest=have or None)
    return "resumed" if have else "new"


# ------------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="measure sizes, print the selection plan, download nothing")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    ap.add_argument("--no-repair-splits", action="store_true",
                    help="do not reassign patients that straddle the official splits; "
                         "pool them instead and build your own splits later")
    args = ap.parse_args()

    load_env_file()
    rng = random.Random(SEED)
    budget = int(DISK_BUDGET_GB * GB)
    BRACS_ROOT.mkdir(parents=True, exist_ok=True)

    ftp = connect()

    root = find_root(ftp)
    if not root:
        sys.exit("[bracs] could not locate the 'Whole Slide Image Set' directory on the server.\n"
                 "        Log in manually and pass the correct path via BRACS_WSI_DIR.")
    root = os.environ.get("BRACS_WSI_DIR", root)
    print(f"[bracs] WSI root: {root}")

    print("[bracs] indexing remote files (measuring sizes, no download yet) ...")
    files = list(tqdm(walk(ftp, root), desc="  indexing", unit="file"))
    svs = {Path(p).stem: (p, s) for p, s in files if p.lower().endswith(".svs")}
    xlsx = find_metadata(ftp)  # BRACS.xlsx sits at the server root, not under BRACS_WSI
    print(f"[bracs] found {len(svs)} .svs ({sum(s for _, s in svs.values())/GB:.1f} GB total) "
          f"and {len(xlsx)} metadata file(s) at the server root")

    if not xlsx:
        sys.exit("[bracs] no .xlsx metadata found at the server root; cannot map slides to "
                 "patients, and a patient-disjoint subset is impossible without it.")

    meta_local = BRACS_ROOT / Path(xlsx[0]).name
    if not meta_local.exists():
        with open(meta_local, "wb") as fh:
            ftp.retrbinary(f"RETR {xlsx[0]}", fh.write)
    print(f"[bracs] metadata: {meta_local}")

    meta = parse_metadata(meta_local)
    meta = meta[meta["slide_id"].isin(svs)]
    sizes = {sid: svs[sid][1] for sid in meta["slide_id"]}
    print(f"[bracs] {len(meta)} slides matched to metadata across "
          f"{meta['patient_id'].nunique()} patients")
    print(f"[bracs] class distribution: {dict(meta['label'].value_counts())}")

    disjoint, report = check_patient_disjoint(meta)
    print(f"\n[bracs] official split patient-disjoint? {'YES' if disjoint else 'NO'} -- {report}")
    if not disjoint and not args.no_repair_splits:
        meta, moves = repair_splits(meta)
        print("[bracs] repairing: each straddling patient is reassigned wholly to one split")
        for pid, frm, to, n in moves:
            print(f"[bracs]   patient {pid}: {n} slide(s) moved {frm} -> {to}")
        disjoint, report = check_patient_disjoint(meta)
        print(f"[bracs] after repair, patient-disjoint? {'YES' if disjoint else 'NO'} -- {report}")
        if not disjoint:
            sys.exit("[bracs] repair failed; refusing to proceed with a leaking split.")
    elif not disjoint:
        print("[bracs] WARNING: the official split shares patients across sets and --no-repair-\n"
              "        splits was given. It is NOT safe for patient-disjoint evaluation.\n"
              "        Selecting from the pooled set; build your own grouped splits first.")

    # ---- selection
    plan = []
    if disjoint:
        totals = meta.groupby("official_split").size()
        for split, sub in meta.groupby("official_split"):
            share = totals[split] / totals.sum()
            pats, spent = select_patients(sub, sizes, int(budget * share), MIN_PER_CLASS, rng)
            sel = sub[sub["patient_id"].isin(pats)].assign(split=split)
            plan.append(sel)
            print(f"[bracs]   {split:<6s}: {len(pats):>3d} patients  {len(sel):>3d} slides  "
                  f"{spent/GB:6.1f} GB (budget {budget*share/GB:.1f} GB)")
    else:
        pats, spent = select_patients(meta, sizes, budget, MIN_PER_CLASS, rng)
        plan.append(meta[meta["patient_id"].isin(pats)].assign(split="unassigned"))
        print(f"[bracs]   pooled: {len(pats)} patients  {len(plan[0])} slides  {spent/GB:.1f} GB")

    sel = pd.concat(plan, ignore_index=True)
    sel["bytes"] = sel["slide_id"].map(sizes)
    sel["remote"] = sel["slide_id"].map(lambda s: svs[s][0])
    total = int(sel["bytes"].sum())

    # patient-disjointness of what we are about to download, asserted not assumed
    if disjoint:
        gs = {s: set(g["patient_id"]) for s, g in sel.groupby("split")}
        for a in gs:
            for b in gs:
                if a < b:
                    assert gs[a].isdisjoint(gs[b]), f"selected subset leaks patients {a}/{b}"

    print(f"\n[bracs] SELECTED {len(sel)} slides / {sel['patient_id'].nunique()} patients "
          f"= {total/GB:.1f} GB (budget {DISK_BUDGET_GB:.0f} GB)")
    print(f"[bracs] selected class distribution: {dict(sel['label'].value_counts())}")
    thin = {c: n for c, n in sel['label'].value_counts().items() if n < MIN_PER_CLASS}
    if thin:
        print(f"[bracs] NOTE: classes below MIN_PER_CLASS={MIN_PER_CLASS}: {thin} "
              f"(the cohort itself may not contain more)")

    manifest = BRACS_ROOT / "manifest.csv"
    sel.drop(columns=["remote"]).to_csv(manifest, index=False)
    print(f"[bracs] wrote manifest: {manifest}")

    if args.dry_run:
        print("\n[bracs] --dry-run: nothing downloaded. Re-run without --dry-run to fetch.")
        ftp.quit()
        return

    free = os.statvfs(BRACS_ROOT).f_bavail * os.statvfs(BRACS_ROOT).f_frsize
    if total > free:
        sys.exit(f"[bracs] need {total/GB:.1f} GB but only {free/GB:.1f} GB free on disk.")
    if not args.yes:
        resp = input(f"\nDownload {len(sel)} slides ({total/GB:.1f} GB) to {BRACS_ROOT}? [y/N] ")
        if resp.strip().lower() not in ("y", "yes"):
            print("[bracs] aborted.")
            ftp.quit()
            return

    print(f"\n[bracs] downloading -> {BRACS_ROOT}  (resumable; Ctrl-C is safe)")
    counts = defaultdict(int)
    with tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024,
              desc="overall", position=1) as overall:
        for r in sel.itertuples():
            local = BRACS_ROOT / "wsi" / r.split / r.label / f"{r.slide_id}.svs"
            for attempt in range(3):
                try:
                    counts[download_file(ftp, r.remote, local, r.bytes, overall)] += 1
                    break
                except (ftplib.all_errors, OSError) as e:
                    if attempt == 2:
                        print(f"\n[bracs] FAILED {r.slide_id}: {e}")
                        counts["failed"] += 1
                        break
                    print(f"\n[bracs] retry {r.slide_id} after {type(e).__name__}; reconnecting")
                    try:
                        ftp.quit()
                    except Exception:
                        pass
                    ftp = connect()

    print(f"\n[bracs] done: {dict(counts)}")
    print(f"[bracs] slides under {BRACS_ROOT/'wsi'}, manifest at {manifest}")
    if counts.get("failed"):
        print("[bracs] some slides failed -- re-run to retry them (completed ones are skipped).")
    try:
        ftp.quit()
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(main())
