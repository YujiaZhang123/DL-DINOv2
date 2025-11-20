

import os
from datasets import load_dataset
from PIL import Image
import numpy as np
from tqdm import tqdm


def main():

    # download data from huggingface
    SAVE_ROOT = "./hf_dataset"
    PRETRAIN_DIR = os.path.join(SAVE_ROOT, "pretrain")

    #
    os.makedirs(PRETRAIN_DIR, exist_ok=True)

    print(">>> From HuggingFace Load Dataset tsbpp/fall2025_deeplearning ...")
    ds = load_dataset(
        "tsbpp/fall2025_deeplearning",
        split="pretrain",
        streaming=False
    )

    total = len(ds)
    print(f">>> Dataset contains {total} images")
    print(f">>> Save DIR: {PRETRAIN_DIR}")


    for i, sample in enumerate(tqdm(ds, desc="save pretrain images", ncols=80)):
        img = sample["image"]

        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))

        out_path = os.path.join(PRETRAIN_DIR, f"{i}.jpg")
        img.save(out_path, "JPEG", quality=95)

    print(f">>> Done! All images saves as: {PRETRAIN_DIR}")


if __name__ == "__main__":
    main()
