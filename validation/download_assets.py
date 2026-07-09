"""Pre-download gated TITAN assets. RUN THIS YOURSELF (downloads can take a while).

  source .venv/bin/activate
  python validation/download_assets.py

Requires the HF account behind HF_TOKEN to have accepted the license at
https://huggingface.co/MahmoodLab/TITAN (otherwise these 401).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_hf_token


def main():
    token = load_hf_token()
    print(f"[download] HF token loaded: {'yes' if token else 'NO — will fail on gated repos'}")
    from huggingface_hub import login, hf_hub_download
    if token:
        login(token=token, add_to_git_credential=False)

    # 1) TITAN model weights + remote code (also pulls CONCHv1.5 on model init)
    print("[download] fetching MahmoodLab/TITAN model (trust_remote_code)...")
    from transformers import AutoModel
    AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    print("[download] TITAN model cached.")

    # 2) precomputed TCGA slide embeddings
    print("[download] fetching TCGA_TITAN_features.pkl ...")
    p = hf_hub_download("MahmoodLab/TITAN", filename="TCGA_TITAN_features.pkl")
    print(f"[download] pkl cached at {p}")
    print("[download] DONE.")


if __name__ == "__main__":
    main()
