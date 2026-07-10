"""Shared helpers for TITAN validation scripts."""
import os
import sys
import json
import warnings
from pathlib import Path

# make `import titan` work regardless of cwd / editable-install state
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# quiet the noisy-but-harmless startup messages (keeps tqdm progress bars intact)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
warnings.filterwarnings("ignore", message=".*torchvision.datapoints.*")
warnings.filterwarnings("ignore", message=".*Beta.*")

import torch

try:
    import torchvision
    torchvision.disable_beta_transforms_warning()
except Exception:
    pass

# ---- Paths ----
TITAN_ROOT = Path(__file__).resolve().parent.parent
DATASETS = TITAN_ROOT / "datasets"
CONFIG_TCGA_OT = DATASETS / "config_tcga-ot.yaml"

# CAMELYON16 CONCHv1.5 patch features (h5 files) -- not part of this repo, point this
# at your own local copy. See validation/.env.example.
CAM16_ROOT = Path(os.environ.get("CAMELYON16_ROOT", TITAN_ROOT / "data" / "CAMELYON16"))
CAM16_H5 = CAM16_ROOT / "h5_files" / "conch_v1_5"
CAM16_REF = CAM16_ROOT / "reference.csv"

# Optional .env file to read HF_TOKEN from, if not already set in the environment.
ENV_FILE = Path(os.environ.get("TITAN_ENV_FILE", TITAN_ROOT / ".env"))

# Where phase results/checkpoints get written. Defaults to a repo-local, gitignored folder.
RESULTS_DIR = Path(os.environ.get("VALIDATION_RESULTS_DIR", TITAN_ROOT / "validation" / "results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_env_file(path=None):
    """Load KEY=VALUE lines from the .env file into os.environ (existing vars win).

    Ignores blank lines, `#` comments, and strips optional surrounding quotes.
    """
    path = Path(path) if path else ENV_FILE
    if not path.exists():
        return {}
    loaded = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val
        loaded[key] = val
    return loaded


def load_hf_token():
    """Ensure HF_TOKEN is in the environment, reading it from the .env file if needed."""
    if not os.environ.get("HF_TOKEN"):
        load_env_file()
    token = os.environ.get("HF_TOKEN")
    if token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    return token


def load_titan(device=None):
    """Load the TITAN model (downloads CONCHv1.5 + TITAN weights on first call).

    Fails loudly if CUDA is unavailable — silent CPU fallback on these large slides
    means ~55s/slide and can exhaust system RAM (OOM killer). Set ALLOW_CPU=1 to override.
    """
    from transformers import AutoModel

    if not torch.cuda.is_available() and os.environ.get("ALLOW_CPU") != "1":
        raise RuntimeError(
            "CUDA is not available (torch.cuda.is_available() == False). The NVIDIA "
            "driver may have crashed — run `nvidia-smi` to check. Refusing to run on CPU "
            "(too slow / RAM-unsafe for slide encoding). Set ALLOW_CPU=1 to force CPU.")

    # HF_TOKEN in the environment is picked up automatically by from_pretrained;
    # no explicit login() needed (avoids the noisy token note).
    load_hf_token()
    device = device or get_device()
    model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    model = model.to(device)
    model.eval()
    return model, device


def save_results(name, obj):
    path = RESULTS_DIR / name
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"[saved] {path}")
    return path


# TITAN's slide encoder uses dense O(N^2) attention. On a 24GB GPU, very large slides
# (these 40x CAMELYON16 features reach ~23k patches) exceed memory. We PREEMPTIVELY cap
# patches BEFORE encoding so every forward starts from a clean GPU and never enters the
# OOM/exception path (retrying inside an except keeps the failed tensors alive via the
# traceback -> the retry OOMs too). Empirically verified: MAX_PATCHES fits the worst slide.
MAX_PATCHES = 10000


def encode_slide_from_h5(model, h5_path, device, patch_size_lv0=None, max_patches=MAX_PATCHES):
    """Load CONCHv1.5 patch features+coords from an h5 and return one TITAN slide embedding.

    patch_size_lv0 = distance between adjacent patches at level 0. If not given, read
    from coords.attrs; else fall back to 512 (these CAMELYON16 h5s were extracted with
    --patch_size 512 --step_size 512 --patch_level 0, verified modal coord spacing = 512).
    Slides with > max_patches are deterministically subsampled (seed 0); returns `capped`.
    """
    import h5py

    with h5py.File(h5_path, "r") as f:
        feat_key = "features" if "features" in f else ("feats" if "feats" in f else list(f.keys())[0])
        features = torch.from_numpy(f[feat_key][:])
        coords = torch.from_numpy(f["coords"][:])
        attrs = dict(f["coords"].attrs)
        if patch_size_lv0 is None:
            patch_size_lv0 = int(attrs.get("patch_size_level0", attrs.get("patch_size_lv0", 512)))

    capped = False
    n = features.shape[0]
    if max_patches and n > max_patches:
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(n, generator=g)[:max_patches].sort().values
        features, coords = features[idx], coords[idx]
        capped = True
        print(f"    [cap] {h5_path.name}: {n} -> {max_patches} patches")

    f_ = features.to(device)
    c_ = coords.to(device)
    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast("cuda", torch.float16):
                emb = model.encode_slide_from_patch_features(f_, c_, patch_size_lv0).float().cpu()
        else:
            emb = model.encode_slide_from_patch_features(f_, c_, patch_size_lv0).float().cpu()
    del f_, c_
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return emb, patch_size_lv0, capped
