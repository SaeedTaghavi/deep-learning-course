# Deep learning course

A hands-on course in PyTorch, one architecture per chapter. Each chapter starts from plain training and test scripts, then wraps the same model in a small web app, first on the CPU and then on an NVIDIA GPU, so you see the model, the training loop and what it takes to put a model in front of users.

## Chapters

| # | Chapter | Topic |
|---|---|---|
| 01 | [LeNet-5 on MNIST](01-lenet-mnist) | the first convolutional network: reading handwritten digits |
| 02 | AlexNet | coming |
| 03 | GoogLeNet | coming |
| 04 | ResNet | coming |
| 05 | LSTM | coming |
| 06 | GANs | coming |
| 07 | Diffusion models | coming |

## How each chapter is organised

- **Scripts** (`*-scripts-cpu`): `train.py` and `test.py`, nothing else. Start here.
- **CPU web app** (`*-app-cpu`): the trained model behind a web page, in Docker.
- **GPU web app** (`*-app-gpu`): the same app, trained and served on an NVIDIA GPU.

Each folder has its own README with setup steps and things to try.

## What you need

- Python 3.10 or newer and PyTorch for the scripts.
- [Docker](https://www.docker.com/products/docker-desktop/) for the web apps.
- An NVIDIA GPU with a current driver for the GPU versions. Everything else runs on a CPU.

Datasets are downloaded automatically the first time you run a chapter, and are not stored in this repository.
