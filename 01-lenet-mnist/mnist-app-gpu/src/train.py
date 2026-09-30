"""
train.py -- train LeNet-5 on MNIST on the GPU and publish the weights for the web app.

Motivation
----------
mnist-app-cpu installs the CPU-only torch wheel, so its trainer can never see a
GPU even though it asks for CUDA. This is the GPU counterpart: it runs in a
CUDA-enabled container on the RTX 4070 and writes the weights to the volume the
web app reads from.

On a GPU the usual DataLoader is the bottleneck for MNIST: it decodes one PIL
image at a time on the CPU and copies each batch to the card. MNIST is small
(60k x 28 x 28 as float32 is about 190 MB), so the whole training set is moved
to GPU memory once and mini-batches are sliced from it with a GPU permutation.
Mild random affine augmentation (rotation, scale, shift) is also done on the GPU,
so the model copes better with hand-drawn or photographed digits that are not
perfectly MNIST-like.

What it measures
----------------
- Mean training loss and wall-clock time for every epoch.
- Accuracy on the 10k held-out MNIST test images after every epoch.
- Writes <shared>/lenet_mnist.pth (atomically, so the web app never reads a
  half-written file) and <shared>/lenet_mnist.json with test accuracy, epochs,
  device and training time, which the web app reports on /health.

Usage
-----
Normal route, through docker compose (from the mnist-app-gpu folder):
    docker compose up --build                        # trains once, then serves
    docker compose run --rm -e FORCE_RETRAIN=1 trainer
    docker compose run --rm -e FORCE_RETRAIN=1 -e EPOCHS=20 -e AUGMENT=0 trainer

Directly, on any machine with a CUDA build of torch:
    python train.py --epochs 10 --batch-size 256 --shared-dir ./shared_data --force
"""
import argparse
import json
import math
import os
import time
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST

from lenet import LeNet, describe_device, pick_device

MODEL_NAME = "lenet_mnist.pth"
META_NAME = "lenet_mnist.json"


def env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(int(default))).strip().lower() in ("1", "true", "yes")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", 10)))
    p.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", 256)))
    p.add_argument("--lr", type=float, default=float(os.environ.get("LR", 1e-3)))
    p.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 0)))
    p.add_argument("--shared-dir", default=os.environ.get("SHARED_DIR", "/shared_data"))
    p.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "/workspace/data"))
    p.add_argument("--force", action="store_true", default=env_bool("FORCE_RETRAIN", False),
                   help="retrain even if a model already exists")
    p.add_argument("--no-augment", dest="augment", action="store_false",
                   default=env_bool("AUGMENT", True))
    p.add_argument("--allow-cpu", dest="require_gpu", action="store_false",
                   default=env_bool("REQUIRE_GPU", True))
    return p.parse_args()


def load_split(root: str, train: bool, device: torch.device):
    """Return the whole MNIST split as (N,1,28,28) float in [0,1] and (N,) labels, on device."""
    ds = MNIST(root=root, train=train, download=True)
    x = ds.data.unsqueeze(1).to(device, dtype=torch.float32).div_(255.0)
    y = ds.targets.to(device)
    return x, y


def random_affine(x: torch.Tensor, max_rot_deg=12.0, scale=(0.85, 1.1), max_shift_px=2.5):
    """Random rotation / scale / translation per image, done on the GPU."""
    n, dev = x.shape[0], x.device
    ang = (torch.rand(n, device=dev) * 2 - 1) * math.radians(max_rot_deg)
    s = torch.empty(n, device=dev).uniform_(*scale)
    # affine_grid works in normalised [-1, 1] coordinates: 1 px = 2 / 28
    t = (torch.rand(n, 2, device=dev) * 2 - 1) * (max_shift_px * 2 / x.shape[-1])
    cos, sin = torch.cos(ang) / s, torch.sin(ang) / s
    theta = torch.stack(
        [torch.stack([cos, -sin, t[:, 0]], 1), torch.stack([sin, cos, t[:, 1]], 1)], 1
    )
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


@torch.inference_mode()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch: int = 4096) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(y), batch):
        correct += (model(x[i:i + batch]).argmax(1) == y[i:i + batch]).sum().item()
    return correct / len(y)


def save_atomically(obj, path: str, as_json: bool = False):
    tmp = path + ".tmp"
    if as_json:
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2)
    else:
        torch.save(obj, tmp)
    os.replace(tmp, path)  # atomic on the same filesystem


def main():
    args = parse_args()
    model_path = os.path.join(args.shared_dir, MODEL_NAME)
    meta_path = os.path.join(args.shared_dir, META_NAME)

    if os.path.exists(model_path) and not args.force:
        print(f"Model already at {model_path}; skipping training "
              f"(set FORCE_RETRAIN=1 or pass --force to retrain).")
        return

    device = pick_device(args.require_gpu)
    dev_info = describe_device(device)
    print("Device:", dev_info)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    x_train, y_train = load_split(args.data_dir, True, device)
    x_test, y_test = load_split(args.data_dir, False, device)
    n = len(y_train)
    print(f"Loaded MNIST onto {device}: train {tuple(x_train.shape)}, test {tuple(x_test.shape)}")

    model = LeNet().to(device)
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Training {args.epochs} epochs, batch {args.batch_size}, "
          f"lr {args.lr}, augment={args.augment}")
    t_start = time.perf_counter()
    acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        perm = torch.randperm(n, device=device)
        loss_sum = torch.zeros((), device=device)  # accumulate on GPU, no per-step sync
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb, yb = x_train[idx], y_train[idx]
            if args.augment:
                xb = random_affine(xb)
            loss = loss_fn(model(xb), yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach() * len(yb)
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        acc = evaluate(model, x_test, y_test)
        print(f"Epoch {epoch:2d}/{args.epochs} | loss {loss_sum.item() / n:.4f} "
              f"| test acc {acc * 100:.2f}% | {dt:.2f} s")

    total = time.perf_counter() - t_start
    os.makedirs(args.shared_dir, exist_ok=True)
    save_atomically(model.state_dict(), model_path)
    save_atomically({
        "test_accuracy": round(acc, 4),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "augment": args.augment,
        "train_seconds": round(total, 1),
        "trained_on": dev_info,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, meta_path, as_json=True)
    print(f"Done in {total:.1f} s. Test accuracy {acc * 100:.2f}%. Saved {model_path}")


if __name__ == "__main__":
    main()
