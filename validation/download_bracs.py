"""Download BRACS (breast carcinoma subtyping, 7 classes, CC0) with patient-disjoint splits.

Two image sets are available; `--set roi` is the default and the recommended one.

  roi  4,539 ROIs (.png), 51.8 GB, 387 parent WSIs, 151 patients.
       Fits entirely on disk, so there is no budget subsetting and therefore no selection
       bias. Its official train/val/test split IS patient-disjoint (verified), and classes
       are near-balanced in every split (test: 79-85 ROIs per class). TITAN encodes an ROI
       as a small slide -- the same way the paper's own TCGA-UT-8K evaluation works.

  wsi  547 full slides (.svs), 984.1 GB, 189 patients, mean 1.93 GB/slide.
       Far too large to take whole. A DISK_BUDGET_GB subset is selected at PATIENT
       granularity (BRACS averages ~2.9 slides/patient; slide-level subsetting would put a
       patient on both sides of a split). Caveat: the greedy selector prefers cheap patients,
       so the subset skews toward slides with less tissue, and at 150 GB it yields only ~18
       test queries -- too few to measure a retrieval effect.

Access
------
The BRACS FTP server is ANONYMOUS-ONLY. Registering on the website gates the portal page
that reveals the hostname, not the FTP server, so only BRACS_FTP_HOST is required.
(BRACS_FTP_USER/PASS are optional; if set they are tried before anonymous.)

The server is vsFTPd: no MLSD, and it demands TLS session reuse on the data channel -- stock
ftplib.FTP_TLS logs in fine and then fails every transfer with "522 ... session reuse
required". Both are handled below.

Split integrity
---------------
The WSI split shipped with BRACS is NOT patient-disjoint: patient 67 straddles training
(3 PB slides) and validation (2 N slides); testing is clean. So the leak corrupts
hyperparameter selection, not test evaluation. `repair_splits()` reassigns any straddling
patient wholly to its majority split, and the script refuses to proceed if a leak survives.
The ROI split needs no repair, but is checked all the same rather than trusted.

Usage
-----
    # put the host in .env:  BRACS_FTP_HOST=histoimage.na.icar.cnr.it
    python validation/download_bracs.py --dry-run          # plan only, downloads nothing
    python validation/download_bracs.py                    # ROI set (51.8 GB)
    python validation/download_bracs.py --set wsi          # budgeted WSI subset

Resumable: a file already on disk with the exact remote byte size is skipped; a partial file
resumes via FTP REST. Ctrl-C is safe; re-running never re-fetches completed files.

Env: BRACS_FTP_HOST, BRACS_ROOT, DISK_BUDGET_GB (default 150), MIN_PER_CLASS (default 5), SEED.
"""
import argparse
import ftplib
import os
import random
import re
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
CLASSES = ["N", "PB", "UDH", "FEA", "ADH", "DCIS", "IC"]

# Remote layout, per set:
#   wsi: BRACS_WSI/<split>/Group_<G>/Type_<LABEL>/BRACS_<id>.svs
#   roi: BRACS_RoI/latest_version/<split>/<n>_<LABEL>/BRACS_<wsi>_<LABEL>_<n>.png
SETS = {
    "wsi": {"dir": "BRACS_WSI", "ext": ".svs"},
    "roi": {"dir": "BRACS_RoI/latest_version", "ext": ".png"},
}
METADATA_XLSX = "BRACS.xlsx"  # lives at the SERVER ROOT, not inside the image dirs


# --------------------------------------------------------------------------- FTP


class ReusedSslFTP_TLS(ftplib.FTP_TLS):
    """FTP_TLS that reuses the control channel's TLS session on the data channel.

    vsFTPd with `require_ssl_reuse=YES` (this server) rejects data connections that negotiate
    a fresh TLS session: "522 SSL connection failed: session reuse required".
    """

    def ntransfercmd(self, cmd, rest=None):
        conn, size = ftplib.FTP.ntransfercmd(self, cmd, rest)
        if self._prot_p:
            conn = self.context.wrap_socket(conn, server_hostname=self.host,
                                            session=self.sock.session)
        return conn, size


def _try_login(host, user, pw, use_tls):
    """One connect+login attempt, VALIDATED by a real data-channel round-trip.

    Logging in is not proof of a working connection: TLS data-channel faults only appear on
    the first transfer. The NLST here forces that failure now, so connect() can fall back.
    """
    ftp = ReusedSslFTP_TLS(host, timeout=60) if use_tls else ftplib.FTP(host, timeout=60)
    try:
        ftp.login(user, pw)
        if use_tls:
            ftp.prot_p()
        ftp.set_pasv(True)
        ftp.voidcmd("TYPE I")  # binary mode; SIZE is only valid here
        ftp.nlst()
        return ftp
    except Exception:
        try:
            ftp.close()
        except Exception:
            pass
        raise


def connect():
    host = os.environ.get("BRACS_FTP_HOST")
    if not host:
        sys.exit("[bracs] Missing BRACS_FTP_HOST. Set it in .env (see .env.example).\n"
                 "        The BRACS FTP server allows anonymous access; no username needed.")

    user, pw = os.environ.get("BRACS_FTP_USER"), os.environ.get("BRACS_FTP_PASS")
    attempts = ([(user, pw, "supplied credentials")] if user and pw else []) + \
               [("anonymous", "anonymous@", "anonymous")]

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
                    break  # credentials can never work here; go straight to anonymous
    sys.exit(f"[bracs] could not log in to {host}: {last}")


def _parse_list_line(line):
    """Parse one `ls -l` style LIST line -> (name, size_or_None). None size => directory."""
    parts = line.split(maxsplit=8)
    if len(parts) < 9:
        return None, None
    name = parts[8]
    if name in (".", ".."):
        return None, None
    if line[0] == "d":
        return name, None
    if line[0] == "l":  # symlink: "name -> target"
        return name.split(" -> ")[0], None
    try:
        return name, int(parts[4])
    except ValueError:
        return None, None


def listdir(ftp, path=""):
    """[(full_path, size_or_None)] for one directory; None size => directory.

    This server has no MLSD. NLST+SIZE would cost a round-trip per entry (~1.6 files/s over
    this link); one LIST per directory already carries type and size for every entry, which
    takes a full 547-slide index from ~6 min to ~45 s. NLST+SIZE remains a last resort.
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
        pass

    lines = []
    try:
        ftp.retrlines(f"LIST {path}" if path else "LIST", lines.append)
        out = []
        for line in lines:
            name, size = _parse_list_line(line)
            if name is not None:
                out.append((f"{path.rstrip('/')}/{name}" if path else name, size))
        if out or not lines:
            return out
    except (ftplib.error_perm, ftplib.error_proto, ftplib.error_temp):
        pass

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


def fetch(ftp, remote, local):
    local.parent.mkdir(parents=True, exist_ok=True)
    with open(local, "wb") as fh:
        ftp.retrbinary(f"RETR {remote}", fh.write)
    return local


# --------------------------------------------------------------- metadata parsing


def _pick_column(cols, *keywords):
    for c in cols:
        if any(k in str(c).strip().lower() for k in keywords):
            return c
    return None


def parse_metadata(xlsx_path):
    """BRACS.xlsx sheet 'WSI_Information': WSI Filename | Patient Id | RoI | WSI label | Set.
    Column names are undocumented, so match case-insensitively by keyword."""
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
        "official_split": (df[c_set].astype(str).str.strip().str.lower() if c_set else "unknown"),
    })
    return out.dropna(subset=["slide_id", "patient_id"]).drop_duplicates("slide_id")


def build_items(which, files, wsi_meta, root):
    """Turn the remote file listing into one row per downloadable item.

    wsi: identity/label/split all come from the .xlsx (authoritative, and the only place
         patient IDs exist). Remote paths are matched to it by filename stem.
    roi: split and label come from the ROI directory tree (the ROI split is its OWN split and
         differs from the WSI split); the patient comes from the parent WSI via the .xlsx.
    """
    wsi2pat = dict(zip(wsi_meta.slide_id, wsi_meta.patient_id))

    if which == "wsi":
        remote = {Path(p).stem: (p, s) for p, s in files}
        m = wsi_meta[wsi_meta.slide_id.isin(remote)].copy()
        m["item_id"] = m.slide_id
        m["remote"] = m.slide_id.map(lambda s: remote[s][0])
        m["bytes"] = m.slide_id.map(lambda s: remote[s][1])
        return m[["item_id", "patient_id", "label", "official_split", "remote", "bytes"]]

    rows = []
    for p, size in files:
        rel = p[len(root):].strip("/").split("/")   # <split>/<n>_<LABEL>/<file>.png
        if len(rel) < 3:
            continue
        split, class_dir, fname = rel[0], rel[1], rel[-1]
        match = re.match(r"(BRACS_\d+)", fname)
        if not match:
            continue
        parent = match.group(1)
        rows.append({"item_id": Path(fname).stem, "patient_id": wsi2pat.get(parent),
                     "label": class_dir.split("_", 1)[-1], "official_split": split,
                     "remote": p, "bytes": size, "parent_wsi": parent})
    df = pd.DataFrame(rows)
    unmapped = int(df.patient_id.isna().sum())
    if unmapped:
        sys.exit(f"[bracs] {unmapped} ROIs have no parent WSI in the metadata; cannot "
                 f"guarantee patient-disjoint splits. Refusing to continue.")
    return df


def check_patient_disjoint(meta):
    """Is the split patient-disjoint? Returns (bool, report)."""
    if (meta["official_split"] == "unknown").all():
        return False, "no split column found in metadata"
    groups = {s: set(g["patient_id"]) for s, g in meta.groupby("official_split")}
    names = sorted(groups)
    overlaps = {f"{a}&{b}": sorted(groups[a] & groups[b])
                for i, a in enumerate(names) for b in names[i + 1:] if groups[a] & groups[b]}
    if overlaps:
        return False, "; ".join(f"{k}: {v}" for k, v in overlaps.items())
    return True, "train/val/test patient sets are pairwise disjoint"


def repair_splits(meta):
    """Assign each straddling patient wholly to the split where it holds the most items
    (ties -> the larger split). Returns (meta, moves)."""
    meta = meta.copy()
    rank = {s: i for i, s in enumerate(["training", "train", "testing", "test",
                                        "validation", "val"])}
    moves = []
    for pid, g in meta.groupby("patient_id"):
        splits = set(g["official_split"])
        if len(splits) <= 1:
            continue
        counts = g["official_split"].value_counts()
        keep = sorted(counts.index, key=lambda s: (-counts[s], rank.get(s, 99)))[0]
        moves += [(pid, s, keep, int(counts[s])) for s in splits - {keep}]
        meta.loc[meta["patient_id"] == pid, "official_split"] = keep
    return meta, moves


# ------------------------------------------------------------------- selection


def select_patients(meta, budget_bytes, min_per_class, rng):
    """Greedy, all-or-nothing per patient, under a byte budget.

    Pass 1 secures class coverage (rarest class first, cheapest qualifying patient first) so
    small classes are not starved. Pass 2 spends the remainder on the best items-per-byte.
    """
    by_pat = defaultdict(list)
    for r in meta.itertuples():
        by_pat[r.patient_id].append(r)

    cost = {p: sum(r.bytes for r in rows) for p, rows in by_pat.items()}
    labels_of = {p: [r.label for r in rows] for p, rows in by_pat.items()}
    global_counts = defaultdict(int)
    for rows in by_pat.values():
        for r in rows:
            global_counts[r.label] += 1

    chosen, spent, class_counts = set(), 0, defaultdict(int)
    for cls in sorted(global_counts, key=lambda c: global_counts[c]):     # rarest first
        for p in sorted((p for p in by_pat if p not in chosen and cls in labels_of[p]),
                        key=lambda p: cost[p]):                            # cheapest first
            if class_counts[cls] >= min_per_class:
                break
            if spent + cost[p] > budget_bytes:
                continue
            chosen.add(p)
            spent += cost[p]
            for lab in labels_of[p]:
                class_counts[lab] += 1

    remaining = [p for p in by_pat if p not in chosen]
    rng.shuffle(remaining)  # deterministic tie-break among equal densities
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

    with open(local, "ab" if have else "wb") as fh, tqdm(
            total=size, initial=have, unit="B", unit_scale=True, unit_divisor=1024,
            desc=f"  {local.name[:30]:<30}", leave=False) as bar:
        def cb(chunk):
            fh.write(chunk)
            bar.update(len(chunk))
            overall.update(len(chunk))
        ftp.retrbinary(f"RETR {remote}", cb, blocksize=1 << 20, rest=have or None)
    return "resumed" if have else "new"


# ------------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", choices=sorted(SETS), default="roi", dest="which",
                    help="roi (4,539 ROIs, 51.8 GB, fits whole) or wsi (984 GB, budgeted subset)")
    ap.add_argument("--dry-run", action="store_true",
                    help="measure sizes, print the plan, download nothing")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    ap.add_argument("--no-repair-splits", action="store_true",
                    help="do not reassign patients that straddle splits; pool them instead")
    args = ap.parse_args()

    load_env_file()
    rng = random.Random(SEED)
    budget = int(DISK_BUDGET_GB * GB)
    dest = BRACS_ROOT / args.which
    BRACS_ROOT.mkdir(parents=True, exist_ok=True)

    ftp = connect()
    root = os.environ.get("BRACS_DIR", SETS[args.which]["dir"])
    ext = SETS[args.which]["ext"]
    print(f"[bracs] set={args.which}  remote root={root}")

    # metadata first: without patient IDs a patient-disjoint split is impossible
    meta_local = BRACS_ROOT / METADATA_XLSX
    if not meta_local.exists():
        print(f"[bracs] fetching {METADATA_XLSX} from the server root ...")
        fetch(ftp, METADATA_XLSX, meta_local)
    wsi_meta = parse_metadata(meta_local)

    print("[bracs] indexing remote files (one LIST per directory; no download yet) ...")
    files = [(p, s) for p, s in tqdm(walk(ftp, root), desc="  indexing", unit="file")
             if p.lower().endswith(ext)]
    total_remote = sum(s for _, s in files)
    print(f"[bracs] {len(files)} {ext} files, {total_remote/GB:.1f} GB on the server")

    meta = build_items(args.which, files, wsi_meta, root)
    print(f"[bracs] {len(meta)} items / {meta.patient_id.nunique()} patients")
    print(f"[bracs] classes: {dict(meta.label.value_counts())}")
    print(f"[bracs] splits:  {dict(meta.official_split.value_counts())}")

    disjoint, report = check_patient_disjoint(meta)
    print(f"\n[bracs] official split patient-disjoint? {'YES' if disjoint else 'NO'} -- {report}")
    if not disjoint and not args.no_repair_splits:
        meta, moves = repair_splits(meta)
        print("[bracs] repairing: each straddling patient is reassigned wholly to one split")
        for pid, frm, to, n in moves:
            print(f"[bracs]   patient {pid}: {n} item(s) moved {frm} -> {to}")
        disjoint, report = check_patient_disjoint(meta)
        print(f"[bracs] after repair, patient-disjoint? {'YES' if disjoint else 'NO'} -- {report}")
        if not disjoint:
            sys.exit("[bracs] repair failed; refusing to proceed with a leaking split.")
    elif not disjoint:
        print("[bracs] WARNING: splits share patients and --no-repair-splits was given.\n"
              "        NOT safe for patient-disjoint evaluation; build your own splits.")

    # ---- selection: take everything if it fits, else budget per split at patient granularity
    need = int(meta.bytes.sum())
    if need <= budget:
        sel = meta.copy()
        print(f"\n[bracs] full set is {need/GB:.1f} GB <= budget {DISK_BUDGET_GB:.0f} GB "
              f"-> taking ALL items (no subsetting, so no selection bias)")
    else:
        print(f"\n[bracs] full set is {need/GB:.1f} GB > budget {DISK_BUDGET_GB:.0f} GB "
              f"-> selecting whole patients per split")
        print("[bracs] NOTE: the greedy selector prefers cheap patients, so the subset skews "
              "toward items with less tissue. Treat it as biased.")
        parts, totals = [], meta.groupby("official_split").size()
        for split, sub in meta.groupby("official_split"):
            share = totals[split] / totals.sum()
            pats, spent = select_patients(sub, int(budget * share), MIN_PER_CLASS, rng)
            got = sub[sub.patient_id.isin(pats)]
            parts.append(got)
            print(f"[bracs]   {split:<11s}: {len(pats):>3d} patients  {len(got):>4d} items  "
                  f"{spent/GB:6.1f} GB (budget {budget*share/GB:.1f} GB)")
        sel = pd.concat(parts, ignore_index=True)

    # assert, don't assume, that what we are about to download is patient-disjoint
    gs = {s: set(g.patient_id) for s, g in sel.groupby("official_split")}
    for a in gs:
        for b in gs:
            if a < b and not gs[a].isdisjoint(gs[b]):
                sys.exit(f"[bracs] selected subset leaks patients between {a} and {b}")

    total = int(sel.bytes.sum())
    print(f"\n[bracs] SELECTED {len(sel)} items / {sel.patient_id.nunique()} patients "
          f"= {total/GB:.1f} GB")
    pivot = sel.pivot_table(index="label", columns="official_split", aggfunc="size", fill_value=0)
    print(f"[bracs] class x split:\n{pivot.to_string()}")
    thin = pivot[pivot.min(axis=1) < MIN_PER_CLASS]
    if len(thin):
        print(f"[bracs] NOTE: these classes fall below MIN_PER_CLASS={MIN_PER_CLASS} in some "
              f"split:\n{thin.to_string()}")

    manifest = BRACS_ROOT / f"manifest_{args.which}.csv"
    sel.drop(columns=["remote"]).to_csv(manifest, index=False)
    print(f"[bracs] wrote manifest: {manifest}")

    if args.dry_run:
        print("\n[bracs] --dry-run: nothing downloaded. Re-run without --dry-run to fetch.")
        ftp.quit()
        return

    st = os.statvfs(BRACS_ROOT)
    free = st.f_bavail * st.f_frsize
    if total > free:
        sys.exit(f"[bracs] need {total/GB:.1f} GB but only {free/GB:.1f} GB free.")
    if not args.yes:
        if input(f"\nDownload {len(sel)} items ({total/GB:.1f} GB) to {dest}? [y/N] "
                 ).strip().lower() not in ("y", "yes"):
            print("[bracs] aborted.")
            ftp.quit()
            return

    print(f"\n[bracs] downloading -> {dest}  (resumable; Ctrl-C is safe)")
    counts = defaultdict(int)
    with tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024,
              desc="overall", position=1) as overall:
        for r in sel.itertuples():
            local = dest / r.official_split / r.label / Path(r.remote).name
            for attempt in range(3):
                try:
                    counts[download_file(ftp, r.remote, local, r.bytes, overall)] += 1
                    break
                except (ftplib.all_errors, OSError) as e:
                    if attempt == 2:
                        print(f"\n[bracs] FAILED {r.item_id}: {e}")
                        counts["failed"] += 1
                        break
                    print(f"\n[bracs] retry {r.item_id} after {type(e).__name__}; reconnecting")
                    try:
                        ftp.quit()
                    except Exception:
                        pass
                    ftp = connect()

    print(f"\n[bracs] done: {dict(counts)}")
    print(f"[bracs] files under {dest}, manifest at {manifest}")
    if counts.get("failed"):
        print("[bracs] some items failed -- re-run to retry (completed ones are skipped).")
    try:
        ftp.quit()
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(main())
