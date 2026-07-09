"""Recon: read ONLY the label/patient columns of every TCGA-UT-8K shard (cheap, no images)
to build a class->shard map. Resumable (caches to JSON). Run with the proxy exported."""
import sys, json, time
sys.path.insert(0, ".")
from common import load_hf_token, RESULTS_DIR
load_hf_token()
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, HfApi

OUT = RESULTS_DIR / "ut8k_label_map.json"
BASE = "datasets/MahmoodLab/TCGA-UniformTumor-8K/data/"


def main():
    fs = HfFileSystem()
    api = HfApi()
    files = sorted(f for f in api.list_repo_files("MahmoodLab/TCGA-UniformTumor-8K",
                                                  repo_type="dataset") if f.endswith(".parquet"))
    done = json.load(open(OUT)) if OUT.exists() else {}
    for i, f in enumerate(files):
        shard = f.split("/")[-1]
        if shard in done:
            continue
        for attempt in range(4):
            try:
                with fs.open(BASE + shard) as fh:
                    tab = pq.ParquetFile(fh).read(columns=["cancer", "PATIENT"])
                from collections import Counter
                cls = Counter(tab.column("cancer").to_pylist())
                pats = sorted(set(tab.column("PATIENT").to_pylist()))
                done[shard] = {"n": tab.num_rows, "classes": dict(cls), "patients": pats}
                break
            except Exception as e:
                if attempt == 3:
                    print(f"  !! {shard} failed: {type(e).__name__}"); done[shard] = {"error": True}
                time.sleep(2)
        json.dump(done, open(OUT, "w"))
        print(f"[{i+1}/{len(files)}] {shard}: {done[shard].get('classes', 'ERR')}")
    print(f"[recon] saved {OUT}")


if __name__ == "__main__":
    main()
