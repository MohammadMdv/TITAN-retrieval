"""Download a patient-disjoint, budget-constrained subset of the BRACS WSI set.

BRACS: 547 breast-carcinoma .svs WSIs from ~187 patients, 7 lesion classes
(N, PB, UDH, FEA, ADH, DCIS, IC), CC0. Registration at https://www.bracs.icar.cnr.it/
grants FTP credentials; there is no public mirror and the portal does not publish sizes,
so this script MEASURES every candidate file over FTP before committing to a download.

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
    # 1. register, then put credentials in .env (see .env.example):
    #      BRACS_FTP_HOST=... BRACS_FTP_USER=... BRACS_FTP_PASS=...
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


def connect():
    host = os.environ.get("BRACS_FTP_HOST")
    user = os.environ.get("BRACS_FTP_USER")
    pw = os.environ.get("BRACS_FTP_PASS")
    if not (host and user and pw):
        sys.exit("[bracs] Missing BRACS_FTP_HOST / BRACS_FTP_USER / BRACS_FTP_PASS.\n"
                 "        Register at https://www.bracs.icar.cnr.it/ to obtain FTP credentials,\n"
                 "        then add them to your .env (see .env.example).")
    try:  # prefer TLS when the server offers it; fall back to plain FTP
        ftp = ftplib.FTP_TLS(host, timeout=60)
        ftp.login(user, pw)
        ftp.prot_p()
        print(f"[bracs] connected to {host} over FTPS")
    except (ftplib.error_perm, ssl.SSLError, OSError):
        ftp = ftplib.FTP(host, timeout=60)
        ftp.login(user, pw)
        print(f"[bracs] connected to {host} over plain FTP")
    ftp.set_pasv(True)
    ftp.voidcmd("TYPE I")  # binary mode; SIZE is only valid here
    return ftp


def walk(ftp, path, depth=0):
    """Yield (remote_path, size_bytes) for every file under `path`.

    Uses MLSD where supported (gives type+size directly); otherwise falls back to NLST and
    probes each entry with SIZE (a directory makes SIZE fail, which is how we tell them apart).
    """
    try:
        entries = list(ftp.mlsd(path))
    except (ftplib.error_perm, ftplib.error_proto):
        entries = None

    if entries is not None:
        for name, facts in entries:
            if name in (".", ".."):
                continue
            child = f"{path}/{name}"
            if facts.get("type") == "dir":
                yield from walk(ftp, child, depth + 1)
            elif facts.get("type") == "file":
                size = int(facts["size"]) if "size" in facts else (ftp.size(child) or 0)
                yield child, size
        return

    for child in ftp.nlst(path):
        if child.rstrip("/").split("/")[-1] in (".", ".."):
            continue
        try:
            size = ftp.size(child)
        except ftplib.error_perm:
            size = None
        if size is None:
            yield from walk(ftp, child, depth + 1)
        else:
            yield child, size


def find_root(ftp):
    """Locate the Whole Slide Image Set directory (name/casing varies)."""
    for base in ("", "/"):
        try:
            for name, facts in ftp.mlsd(base or "."):
                if facts.get("type") == "dir" and any(h in name.lower() for h in WSI_DIR_HINTS):
                    return f"{base.rstrip('/')}/{name}"
        except Exception:
            continue
    return None


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
        detail = "; ".join(f"{k}: {len(v)} patients" for k, v in overlaps.items())
        return False, detail
    return True, "train/val/test patient sets are pairwise disjoint"


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
    xlsx = [p for p, _ in files if p.lower().endswith((".xlsx", ".xls"))]
    print(f"[bracs] found {len(svs)} .svs ({sum(s for _, s in svs.values())/GB:.1f} GB total) "
          f"and {len(xlsx)} metadata file(s)")

    if not xlsx:
        sys.exit("[bracs] no .xlsx metadata found under the WSI root; cannot map slides to "
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
    if not disjoint:
        print("[bracs] WARNING: the official split shares patients across sets. It is NOT safe\n"
              "        for patient-disjoint evaluation. Selecting patients from the pooled set;\n"
              "        build your own grouped splits before training.")

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
