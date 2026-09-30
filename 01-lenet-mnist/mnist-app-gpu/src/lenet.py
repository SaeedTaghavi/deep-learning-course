"""
lenet.py -- the LeNet-5 model shared by the trainer and the web app.

Motivation
----------
In mnist-app-cpu the LeNet class is copy-pasted into train.py and main.py. If the
two copies drift apart (a layer size changed in one file only) the web app fails
to load the weights, or loads them into the wrong architecture. Here both
containers import the same file, so there is one definition of the network.

What it measures
----------------
Nothing by itself. It defines the network (2 conv + 2 FC layers, log-softmax
output, 431k parameters) and a helper that picks the compute device.

Usage
-----
    from lenet import LeNet, pick_device
    device = pick_device(require_gpu=True)
    model = LeNet().to(device)

    python lenet.py        # prints the parameter count and the device found
"""
import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self, num_channels: int = 1, classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 20, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
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


def pick_device(require_gpu: bool = True) -> torch.device:
    """Return the CUDA device, or fail loudly instead of silently using the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if require_gpu:
        raise RuntimeError(
            "CUDA is not available inside this container "
            f"(torch {torch.__version__}, built for CUDA {torch.version.cuda}). "
            "Check that the container was started with GPU access "
            "(docker run --gpus all ..., or the deploy.resources block in "
            "docker-compose.yml) and that the NVIDIA driver is new enough for "
            "this CUDA version. Set REQUIRE_GPU=0 to allow a CPU fallback."
        )
    return torch.device("cpu")


def describe_device(device: torch.device) -> dict:
    info = {
        "device": str(device),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    if device.type == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        info.update(
            gpu=props.name,
            compute_capability=f"{props.major}.{props.minor}",
            gpu_memory_gb=round(props.total_memory / 1024**3, 1),
        )
    return info


if __name__ == "__main__":
    m = LeNet()
    print(f"LeNet parameters: {sum(p.numel() for p in m.parameters()):,}")
    print(describe_device(pick_device(require_gpu=False)))
