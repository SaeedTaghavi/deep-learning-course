"""
export_pngs.py -- extract the MNIST images from the raw IDX files as PNG files.

Motivation
----------
torchvision downloads MNIST in its original 1998 format, IDX: all 60,000
training images in one binary file and all 10,000 test images in another. That
is fast for training, but you cannot open it in an image viewer, and you cannot
upload it to the web apps in mnist-app-cpu and mnist-app-gpu, which expect an
ordinary image file. This script writes every image (or a few per digit) as a
separate PNG, sorted into one folder per digit, the same layout as the
mnist-pngs repository.

It reads the IDX format by hand with NumPy, as explained in this folder's
README, so it does not need PyTorch and doubles as a worked example of the format.

What it produces
----------------
    <out>/train/0/00001.png ... <out>/train/9/...
    <out>/test/0/00003.png  ... <out>/test/9/...
    <out>/labels.csv        one line per image: split, index, label, path

The number in each file name is the image's position in the original file, so
test/7/00000.png is the first test image, and its label is 7.

By default the PNGs are exactly the MNIST pixels: 28 x 28, white digit on black.
For uploading to the web apps, two options help:
  --invert   dark digit on white, like ink on paper. Use this for mnist-app-cpu,
             which always inverts uploads and so misreads white-on-black images.
             mnist-app-gpu detects the background and handles both.
  --scale N  enlarge N times (nearest neighbour, so the pixels stay sharp), which
             makes the images easier to see in a file browser and on the page.

Usage
-----
    python export_pngs.py                                   # all 70,000 images -> ./mnist_pngs
    python export_pngs.py --split test --per-digit 5        # 50 test images, 5 of each digit
    python export_pngs.py --split test --per-digit 5 --invert --scale 10 --out mnist_pngs_upload
    python export_pngs.py --data-dir ./data --out /tmp/mnist_pngs

The raw files must already be in <data-dir>/MNIST/raw. Running train.py or
test.py once downloads them.
"""
import argparse
import csv
import gzip
import os
import sys

import numpy as np
from PIL import Image

SPLITS = {
    "train": ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
    "test": ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
}
IMAGES_MAGIC, LABELS_MAGIC = 2051, 2049


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract MNIST images from the raw IDX files as PNGs.")
    p.add_argument("--data-dir", default="./data", help="folder that contains MNIST/raw")
    p.add_argument("--out", default="mnist_pngs", help="output folder")
    p.add_argument("--split", choices=["train", "test", "both"], default="both")
    p.add_argument("--per-digit", type=int, default=None, metavar="N",
                   help="only export the first N images of each digit (default: all)")
    p.add_argument("--invert", action="store_true",
                   help="dark digit on a white background instead of MNIST's white on black")
    p.add_argument("--scale", type=int, default=1, metavar="N",
                   help="enlarge each image N times (default 1, i.e. 28 x 28)")
    return p.parse_args()


def read_idx(path: str, expected_magic: int) -> np.ndarray:
    """Read an IDX file (or its .gz) and return its contents as a uint8 array."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = f.read()
    elif os.path.exists(path + ".gz"):
        with gzip.open(path + ".gz", "rb") as f:
            data = f.read()
    else:
        sys.exit(f"Missing {path}\nRun train.py or test.py once to download MNIST, "
                 f"or point --data-dir at the folder that contains MNIST/raw.")

    # Header: a magic number, then one 4-byte big-endian size per dimension.
    magic = int(np.frombuffer(data[:4], dtype=">u4")[0])
    if magic != expected_magic:
        sys.exit(f"{path}: magic number {magic}, expected {expected_magic}. Not an MNIST file?")
    ndim = magic & 0xFF                       # 3 for images, 1 for labels
    shape = tuple(int(v) for v in np.frombuffer(data[4:4 + 4 * ndim], dtype=">u4"))
    body = np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim)
    return body.reshape(shape)


def export_split(split: str, args, writer) -> int:
    img_file, lbl_file = SPLITS[split]
    raw = os.path.join(args.data_dir, "MNIST", "raw")
    images = read_idx(os.path.join(raw, img_file), IMAGES_MAGIC)
    labels = read_idx(os.path.join(raw, lbl_file), LABELS_MAGIC)
    if len(images) != len(labels):
        sys.exit(f"{split}: {len(images)} images but {len(labels)} labels")

    for d in range(10):
        os.makedirs(os.path.join(args.out, split, str(d)), exist_ok=True)

    written = [0] * 10
    for i, (img, label) in enumerate(zip(images, labels)):
        label = int(label)
        if args.per_digit is not None and written[label] >= args.per_digit:
            if min(written) >= args.per_digit:
                break                         # every digit has enough
            continue
        if args.invert:
            img = 255 - img
        im = Image.fromarray(img)
        if args.scale > 1:
            im = im.resize((28 * args.scale, 28 * args.scale), Image.NEAREST)
        rel = os.path.join(split, str(label), f"{i:05d}.png")
        im.save(os.path.join(args.out, rel))
        writer.writerow([split, i, label, rel.replace(os.sep, "/")])
        written[label] += 1
        n = sum(written)
        if n % 5000 == 0:
            print(f"  {split}: {n:,} images written", flush=True)

    print(f"{split}: {sum(written):,} PNGs  (per digit: {written})")
    return sum(written)


def main():
    args = parse_args()
    if args.scale < 1 or (args.per_digit is not None and args.per_digit < 1):
        sys.exit("--scale and --per-digit must be at least 1")
    splits = ["train", "test"] if args.split == "both" else [args.split]
    os.makedirs(args.out, exist_ok=True)

    style = "dark on white" if args.invert else "white on black (as in MNIST)"
    print(f"Exporting {', '.join(splits)} to {args.out}/  |  {28 * args.scale}x{28 * args.scale} px, {style}")
    with open(os.path.join(args.out, "labels.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "index", "label", "path"])
        total = sum(export_split(s, args, writer) for s in splits)
    print(f"Done: {total:,} images in {args.out}/, labels in {args.out}/labels.csv")


if __name__ == "__main__":
    main()
