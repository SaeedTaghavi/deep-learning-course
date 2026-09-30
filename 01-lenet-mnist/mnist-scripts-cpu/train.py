"""
train.py -- train LeNet-5 on MNIST on the CPU and save the weights.

Motivation
----------
This is the smallest complete version of the LeNet chapter: no Docker, no web app,
no GPU, just the training loop. It is the place to read first, because the two
web apps in this folder (mnist-app-cpu, mnist-app-gpu) wrap exactly this model
and this loop. The question it answers is: what does it take to go from raw
MNIST images to a network that reads handwritten digits at about 99% accuracy,
and how long does that take on an ordinary CPU?

LeNet-5 (LeCun et al., 1998) is the classic small convolutional network:
    conv 5x5, 20 maps -> ReLU -> max-pool 2x2
    conv 5x5, 50 maps -> ReLU -> max-pool 2x2
    fully connected 800 -> 500 -> ReLU -> 10 -> log-softmax
It has about 431k parameters, and all of it is defined in the LeNet class below.

What it measures
----------------
For every epoch: mean training loss, accuracy on the training batches seen in
that epoch, and wall-clock time. At the end it saves the trained weights
(state_dict) to lenet_mnist.pth, which test.py loads. Test-set accuracy is
deliberately left to test.py, so that training and evaluation stay separate steps.

Usage
-----
    python train.py                         # 5 epochs, batch 64, saves ./lenet_mnist.pth
    python train.py --epochs 10 --lr 5e-4
    python train.py --threads 8             # limit the CPU threads PyTorch uses
    python train.py --out weights/lenet.pth --data-dir ./data

MNIST (about 12 MB) is downloaded into --data-dir on the first run.
"""
import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class LeNet(nn.Module):
    def __init__(self, num_channels: int = 1, classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 20, kernel_size=5)   # 1x28x28 -> 20x24x24
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)     # -> 20x12x12
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)             # -> 50x8x8
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)     # -> 50x4x4 = 800
        self.fc1 = nn.Linear(800, 500)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(500, classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.maxpool2(self.relu2(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu3(self.fc1(x))
        return self.logsoftmax(self.fc2(x))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LeNet-5 on MNIST on the CPU.")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=None,
                   help="CPU threads for PyTorch (default: PyTorch decides)")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out", default="lenet_mnist.pth")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    print(f"Device: cpu, {torch.get_num_threads()} threads, torch {torch.__version__}")

    train_set = MNIST(root=args.data_dir, train=True, download=True, transform=ToTensor())
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    model = LeNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"LeNet-5: {n_params:,} parameters | {len(train_set):,} training images")

    loss_fn = nn.NLLLoss()   # the model outputs log-probabilities, so NLL = cross-entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    t_total = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        loss_sum, correct = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            log_probs = model(x)
            loss = loss_fn(log_probs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * len(y)
            correct += (log_probs.argmax(1) == y).sum().item()

        n = len(train_set)
        print(f"Epoch {epoch:2d}/{args.epochs} | loss {loss_sum / n:.4f} "
              f"| train acc {correct / n * 100:.2f}% | {time.perf_counter() - t0:.1f} s")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"Done in {time.perf_counter() - t_total:.1f} s. Weights saved to {args.out}")
    print(f"Next: python test.py --weights {args.out}")


if __name__ == "__main__":
    main()
