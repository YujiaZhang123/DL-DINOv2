# ================================================================
# download.py — Fast HF snapshot download + 8-thread JPG extraction
# ================================================================

import os
import concurrent.futures
from datasets import load_dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
from huggingface_hub import snapshot_download


def save_image(idx, sample, output_dir):
    """Save a single image (used by multithreading executor)."""
    try:
        img = sample["image"]

        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))

        out_path = os.path.join(output_dir, f"{idx}.jpg")
        img.save(out_path, "JPEG", quality=95)
        return True
    except Exception:
        return False


def main():

    SAVE_ROOT = "./hf_dataset"
    SNAPSHOT_ROOT = "./hf_snapshot"
    PRETRAIN_DIR = os.path.join(SAVE_ROOT, "pretrain")

    os.makedirs(PRETRAIN_DIR, exist_ok=True)
    os.makedirs(SNAPSHOT_ROOT, exist_ok=True)

    print(">>> Downloading HF snapshot (Arrow files)...")

    # HF hub shows internal chunk progress; tqdm wraps total=1
    with tqdm(total=1, desc="snapshot_download", ncols=80) as pbar:
        local_path = snapshot_download(
            repo_id="tsbpp/fall2025_deeplearning",
            repo_type="dataset",
            local_dir=SNAPSHOT_ROOT,
            local_dir_use_symlinks=False,
        )
        pbar.update(1)

    print(">>> Snapshot stored at:", local_path)
    print(">>> Loading dataset from local snapshot...")
    ds = load_dataset(
        local_path,
        split="pretrain",
        streaming=False,
        keep_in_memory=False,
    )

    total = len(ds)
    print(f">>> Total images detected in dataset: {total}")
    print(f">>> Output directory: {PRETRAIN_DIR}")

    print(">>> Extracting images with 8 threads...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(save_image, idx, sample, PRETRAIN_DIR)
            for idx, sample in enumerate(ds)
        ]

        # tqdm over completed futures
        for _ in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Extracting images",
            ncols=80
        ):
            pass

    print("\n>>> Extraction complete!")

    final_count = len([
        f for f in os.listdir(PRETRAIN_DIR)
        if f.lower().endswith(".jpg")
    ])

    print(f">>> Total extracted JPG images: {final_count}")
    print(f">>> JPGs saved under: {PRETRAIN_DIR}")


if __name__ == "__main__":
    main()
