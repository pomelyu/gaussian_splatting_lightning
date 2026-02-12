import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataroot", type=str)
    parser.add_argument("--valid_ratio", type=float, default=0.05)
    args = parser.parse_args()

    dataroot = Path(args.dataroot)
    images = [f for f in (dataroot / "images").iterdir() if f.suffix.lower() in [".jpg", ".png"]]
    print(f"gather {len(images)} images")

    n_valid = np.ceil(len(images) * args.valid_ratio).astype(np.int32)
    valid_index = np.linspace(0, len(images), n_valid, dtype=np.int32, endpoint=False)
    train_index = [i for i in range(len(images)) if i not in valid_index]

    with Path(dataroot / "train_image_idx.txt").open("w") as f:
        for i in train_index:
            f.write(f"{i+1:d}\n")
    print(f"add {len(train_index)} images to", dataroot / "train_image_idx.txt")

    with Path(dataroot / "valid_image_idx.txt").open("w") as f:
        for i in valid_index:
            f.write(f"{i+1:d}\n")
    print(f"add {len(valid_index)} images to", dataroot / "valid_image_idx.txt")

if __name__ == "__main__":
    main()
