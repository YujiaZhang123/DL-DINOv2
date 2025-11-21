# ================================================================
# download.py — snapshot + zipfi;e
# ================================================================

import os
import zipfile
from glob import glob
from tqdm import tqdm
from huggingface_hub import snapshot_download
from concurrent.futures import ThreadPoolExecutor, as_completed

HF_REPO = "tsbpp/fall2025_deeplearning"
MAX_THREADS = 4    
MAX_ZIP_WORKERS = 5 

def extract_single_zip(zip_path, out_dir):
    zname = os.path.basename(zip_path)
    print(f"\n>>> Extracting {zname} ...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        pending = [m for m in members if not os.path.exists(os.path.join(out_dir, m))]

        if len(pending) == 0:
            print(f"    {zname}: already extracted, skipping.")
            return len(members)

        def extract_one(m):
            zf.extract(m, out_dir)
            return True

        with ThreadPoolExecutor(max_workers=MAX_THREADS) as ex:
            futures = [ex.submit(extract_one, m) for m in pending]

            for _ in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"{zname}",
                unit="file",
                ncols=80,
            ):
                pass

    return len(members)


def main():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HOME"] = "/workspace/.cache/hf"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    SNAPSHOT_ROOT = "./hf_snapshot"
    OUT_DIR = "./hf_dataset/train"

    os.makedirs(SNAPSHOT_ROOT, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    print(">>> Downloading HF snapshot ...")

    local_path = snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        local_dir=SNAPSHOT_ROOT,
        local_dir_use_symlinks=False,
    )
    print(f"Snapshot stored at: {local_path}")

    # -----------------------------------------------------------
    # Locate ZIP files
    # -----------------------------------------------------------
    zip_files = sorted(glob(os.path.join(local_path, "cc3m_96px_part*.zip")))

    if not zip_files:
        raise RuntimeError("No ZIP files found in snapshot! Something is wrong.")

    print("\n>>> Found ZIP files:")
    for z in zip_files:
        print("   -", os.path.basename(z))

    # -----------------------------------------------------------
    # Parallel extraction of each ZIP
    # -----------------------------------------------------------
    print("\n>>> Starting parallel ZIP extraction ...")

    with ThreadPoolExecutor(max_workers=MAX_ZIP_WORKERS) as executor:
        results = list(executor.map(lambda z: extract_single_zip(z, OUT_DIR), zip_files))

    # -----------------------------------------------------------
    # Final JPG count
    # -----------------------------------------------------------
    print("\n>>> Counting extracted images ...")
    total_jpg = sum(
        1 for f in os.listdir(OUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    print(f"\n>>> Extraction complete!")
    print(f">>> Total extracted images: {total_jpg}")
    print(f">>> Images saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
