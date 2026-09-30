"""
test.py -- evaluate the trained LeNet-5 on the MNIST test set, or classify your own image.

Motivation
----------
Training accuracy says how well the network fits the images it learned from.
The real question is how well it reads digits it has never seen. This script
answers that on the 10,000 MNIST test images, which train.py never touches, and
shows where the network fails: which digits it gets wrong most often and which
digits it confuses with which (4 with 9, 3 with 5, 7 with 2 are typical).

With --image it also classifies a single picture of your own, which is where
the gap between MNIST and real handwriting shows up. MNIST digits are white on
black, centred and about 20x20 pixels inside a 28x28 frame, so the image is
converted to that format first, the same way the web apps do it.

What it measures
----------------
Test mode (default):
  - overall accuracy and the number of misclassified images
  - accuracy per digit
  - the 10x10 confusion matrix (rows: true digit, columns: predicted digit)
  - the most frequent confusions
  - optional: a PNG grid of misclassified test images (--save-errors, needs matplotlib)
Image mode (--image): the predicted digit and the probability of every digit.

Usage
-----
    python test.py                                  # evaluate ./lenet_mnist.pth
    python test.py --weights weights/lenet.pth
    python test.py --save-errors errors.png         # also save the misclassified images
    python test.py --image my_digit.png             # classify one image of your own
"""
import argparse
import sys

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from train import LeNet   # same model definition as training


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate LeNet-5 on MNIST (CPU).")
    p.add_argument("--weights", default="lenet_mnist.pth")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--image", help="classify this image file instead of running the test set")
    p.add_argument("--save-errors", metavar="PNG",
                   help="save a grid of misclassified test images to this file")
    return p.parse_args()


def load_model(path: str) -> LeNet:
    model = LeNet()
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        sys.exit(f"No weights at {path}. Run train.py first.")
    model.load_state_dict(state)
    model.eval()
    return model


# ------------------------------------------------------------------ test set
def evaluate(model: LeNet, args):
    test_set = MNIST(root=args.data_dir, train=False, download=True, transform=ToTensor())
    loader = DataLoader(test_set, batch_size=args.batch_size)

    preds, labels = [], []
    with torch.inference_mode():
        for x, y in loader:
            preds.append(model(x).argmax(1))
            labels.append(y)
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()

    n = len(labels)
    wrong = np.flatnonzero(preds != labels)
    print(f"Test accuracy: {(n - len(wrong)) / n * 100:.2f}%  "
          f"({n - len(wrong):,} / {n:,} correct, {len(wrong)} misclassified)\n")

    cm = np.zeros((10, 10), dtype=int)
    np.add.at(cm, (labels, preds), 1)

    print("Accuracy per digit")
    for d in range(10):
        total = cm[d].sum()
        acc = cm[d, d] / total
        bar = "#" * round(acc * 40)
        print(f"  {d}: {acc * 100:6.2f}%  {bar:<40}  ({total - cm[d, d]} wrong of {total})")

    print("\nConfusion matrix (rows = true digit, columns = predicted digit)")
    print("      " + "".join(f"{d:>6}" for d in range(10)))
    for d in range(10):
        print(f"  {d}   " + "".join(f"{v:>6}" for v in cm[d]))

    off = [(cm[t, p], t, p) for t in range(10) for p in range(10) if t != p and cm[t, p]]
    print("\nMost frequent confusions")
    for count, t, p in sorted(off, reverse=True)[:5]:
        print(f"  true {t} read as {p}: {count} times")

    if args.save_errors:
        save_error_grid(test_set, wrong, preds, labels, args.save_errors)


def save_error_grid(test_set, wrong, preds, labels, path, max_images=64):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n--save-errors needs matplotlib: pip install matplotlib")
        return
    idx = wrong[:max_images]
    cols = 8
    rows = max(1, int(np.ceil(len(idx) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.3, rows * 1.5))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, i in zip(np.atleast_1d(axes).ravel(), idx):
        ax.imshow(test_set.data[i].numpy(), cmap="gray")
        ax.set_title(f"true {labels[i]}, got {preds[i]}", fontsize=8)
    fig.suptitle(f"Misclassified MNIST test images (first {len(idx)} of {len(wrong)})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\nSaved {len(idx)} misclassified images to {path}")


# ------------------------------------------------------------------ one image
def to_mnist_format(img: Image.Image) -> torch.Tensor:
    """Convert any digit picture to a 1x1x28x28 tensor laid out like MNIST."""
    g = np.asarray(ImageOps.exif_transpose(img).convert("L"), dtype=np.float32) / 255.0

    # MNIST is light ink on a dark background: invert if the border is bright.
    border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    if border.mean() > 0.5:
        g = 1.0 - g

    # Contrast stretch: background (median pixel) -> 0, brightest ink -> 1.
    lo, hi = float(np.median(g)), float(g.max())
    if hi - lo < 0.05:
        sys.exit("The image looks blank.")
    g = np.clip((g - lo) / (hi - lo), 0.0, 1.0)

    # Crop to the ink and fit the longest side into 20 px.
    rows = np.flatnonzero((g > 0.25).any(1))
    cols = np.flatnonzero((g > 0.25).any(0))
    crop = g[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    h, w = crop.shape
    s = 20.0 / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    crop = np.asarray(Image.fromarray((crop * 255).astype(np.uint8))
                      .resize((nw, nh), Image.LANCZOS), dtype=np.float32) / 255.0

    # Paste into a 28x28 frame and move the centre of mass to the middle.
    canvas = np.zeros((28, 28), np.float32)
    top, left = (28 - nh) // 2, (28 - nw) // 2
    canvas[top:top + nh, left:left + nw] = crop
    yy, xx = np.indices(canvas.shape)
    m = canvas.sum()
    dy = int(round(13.5 - (yy * canvas).sum() / m))
    dx = int(round(13.5 - (xx * canvas).sum() / m))
    dy = max(-top, min(dy, 28 - (top + nh)))       # never push ink off the frame
    dx = max(-left, min(dx, 28 - (left + nw)))
    canvas = np.roll(canvas, (dy, dx), axis=(0, 1))
    return torch.from_numpy(canvas)[None, None]


def classify(model: LeNet, path: str):
    try:
        img = Image.open(path)
    except (FileNotFoundError, OSError) as e:
        sys.exit(f"Could not open {path}: {e}")
    x = to_mnist_format(img)
    with torch.inference_mode():
        probs = model(x).exp()[0].numpy()
    pred = int(probs.argmax())

    print(f"{path}: predicted digit {pred} ({probs[pred] * 100:.1f}%)\n")
    for d, p in enumerate(probs):
        mark = "  <-" if d == pred else ""
        print(f"  {d}: {p * 100:6.2f}%  {'#' * round(p * 40)}{mark}")
    if probs[pred] < 0.6:
        print("\nLow confidence: the network isn't sure about this one.")


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    model = load_model(args.weights)
    if args.image:
        classify(model, args.image)
    else:
        evaluate(model, args)


if __name__ == "__main__":
    main()
