# ================================================================
# download.py — Fast download of HF snapshot + Extract Arrow to JPG
# Produces:
#   hf_dataset/pretrain/0.jpg
#   hf_dataset/pretrain/1.jpg
# ================================================================

import os
from datasets import load_dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
from huggingface_hub import snapshot_download

def main():

    SAVE_ROOT = "./hf_dataset"
    SNAPSHOT_DIR = "./hf_snapshot"     # where snapshot_download stores Arrow files
    PRETRAIN_DIR = os.path.join(SAVE_ROOT, "pretrain")

    os.makedirs(PRETRAIN_DIR, exist_ok=True)
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    print(">>> Downloading HF snapshot (Arrow files)...")
    local_path = snapshot_download(
        repo_id="tsbpp/fall2025_deeplearning",
        repo_type="dataset",
        local_dir=SNAPSHOT_DIR,
        local_dir_use_symlinks=False
    )
    print(">>> Snapshot downloaded at:", local_path)


    print(">>> Loading dataset from local snapshot (no network access)...")
    ds = load_dataset(
        local_path,
        split="pretrain",
        streaming=False,
        keep_in_memory=False
    )

    total = len(ds)
    print(f">>> Total images in dataset: {total}")
    print(f">>> Output directory: {PRETRAIN_DIR}")

    for i, sample in enumerate(tqdm(ds, desc="Extracting images", ncols=80)):
        img = sample["image"]

        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))

        out_path = os.path.join(PRETRAIN_DIR, f"{i}.jpg")
        img.save(out_path, "JPEG", quality=95)

    print("\n>>> Extraction completed successfully!")
    print(f">>> JPG images saved under: {PRETRAIN_DIR}")


if __name__ == "__main__":
    main()
