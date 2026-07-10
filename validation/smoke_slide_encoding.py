"""Smoke test: load TITAN+CONCHv1.5 and exercise the slide-encoding path on ONE local h5.

Verifies:
  - gated HF access works (model downloads/loads)
  - local CAMELYON16 CONCHv1.5 h5 has features + coords (+ patch_size attr)
  - encode_slide_from_patch_features runs and produces a finite slide embedding
"""
import sys
import h5py
import torch

from common import load_titan, encode_slide_from_h5, CAM16_H5, get_device, save_results


def main():
    device = get_device()
    print(f"[smoke] device = {device}")
    if device.type != "cuda":
        print("[smoke] WARNING: CUDA not available; running on CPU (slow).")

    # inspect one h5 before touching the model
    h5_files = sorted(CAM16_H5.glob("*.h5"))
    assert h5_files, f"no h5 files under {CAM16_H5}"
    sample = h5_files[0]
    with h5py.File(sample, "r") as f:
        keys = list(f.keys())
        coord_attrs = dict(f["coords"].attrs) if "coords" in f else {}
        shapes = {k: f[k].shape for k in keys}
    print(f"[smoke] sample={sample.name} keys={keys} shapes={shapes}")
    print(f"[smoke] coords attrs = {coord_attrs}")

    print("[smoke] loading TITAN (first call downloads gated weights)...")
    model, device = load_titan(device)
    print("[smoke] model loaded. return_conch() sanity...")
    conch, transform = model.return_conch()
    print(f"[smoke] CONCHv1.5 loaded: {type(conch).__name__}")

    emb, psz, _ = encode_slide_from_h5(model, sample, device)
    finite = bool(torch.isfinite(emb).all())
    print(f"[smoke] slide_embedding shape={tuple(emb.shape)} dtype={emb.dtype} "
          f"patch_size_lv0={psz} finite={finite}")
    assert finite, "slide embedding contains NaN/Inf"

    save_results("smoke_slide_encoding.json", {
        "sample": sample.name,
        "h5_keys": keys,
        "coord_attrs": {k: str(v) for k, v in coord_attrs.items()},
        "embedding_shape": list(emb.shape),
        "patch_size_lv0": psz,
        "finite": finite,
    })
    print("[smoke] OK")


if __name__ == "__main__":
    sys.exit(main())
