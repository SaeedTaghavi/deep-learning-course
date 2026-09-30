import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

SHARED_DIR = "/shared_data"
MODEL_PATH = os.path.join(SHARED_DIR, "lenet_mnist.pth")

class LeNet(nn.Module):
    def __init__(self, num_channels=1, classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=20, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=800, out_features=500)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=500, out_features=classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.maxpool2(self.relu2(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu3(self.fc1(x))
        return self.logsoftmax(self.fc2(x))

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset & Loader
    train_dataset = MNIST(root="./data", train=True, download=True, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = LeNet().to(device)
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    epochs = 5
    print("Starting training...")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)

        avg_loss = total_loss / len(train_dataset)
        print(f"Epoch {epoch}/{epochs} | Training Loss: {avg_loss:.4f}")

    os.makedirs(SHARED_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved successfully to {MODEL_PATH}")

if __name__ == "__main__":
    train()