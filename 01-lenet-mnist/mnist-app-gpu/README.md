# mnist-app-gpu

The same web app as [`mnist-app-cpu`](../mnist-app-cpu), with the model trained and served on an NVIDIA GPU inside Docker. It also fixes the limitations listed in the CPU version, so it doubles as a worked example of turning a demo into something more robust.

```
mnist-app-gpu/
  Dockerfile            one CUDA PyTorch image, used by both services
  docker-compose.yml    trainer (runs once) -> webapp (serves :8000), both with GPU access
  requirements.txt
  src/
    lenet.py            model definition, imported by both services
    train.py            trains on the GPU, writes lenet_mnist.pth + lenet_mnist.json
    main.py             FastAPI app, inference on the GPU
    templates/index.html
```

## 1. Check that Docker can see the GPU

You need an NVIDIA GPU, a current NVIDIA driver, and Docker with GPU support. On Windows this means Docker Desktop with the WSL 2 engine. The Windows NVIDIA driver also covers WSL, so don't install a separate driver inside WSL. On Linux, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

If the second command prints your GPU, containers can use it.

**Match PyTorch to your driver.** `nvidia-smi` shows the highest CUDA version your driver supports (top right, "CUDA Version"). By default the image installs the CUDA 12.6 build of PyTorch, which needs CUDA Version 12.6 or higher. With a driver that supports CUDA 13.0 or higher, you can use the latest CUDA 13 build from PyPI instead:

```bash
TORCH_INDEX_URL=https://pypi.org/simple docker compose build --no-cache
```

## 2. Run

```bash
cd mnist-app-gpu
docker compose up --build
```

The first build downloads about 3 GB of CUDA libraries. Then:

1. `trainer` loads all of MNIST into GPU memory, trains 10 epochs and prints loss, test accuracy and seconds per epoch. It should reach about 99.2%. It saves the weights and exits.
2. `webapp` starts only after the trainer has exited successfully, loads the weights onto the GPU and serves http://localhost:8000.

The line at the bottom of the page shows the GPU, the PyTorch and CUDA versions, and the model's test accuracy. Under each prediction it shows the time the forward pass took on the GPU.

To get images to upload, extract some from MNIST with [`export_pngs.py`](../mnist-scripts-cpu#get-png-files-of-the-digits). This app detects the background, so both MNIST's white-on-black images and inverted ones (`--invert`) work.

Later runs skip training, because the model is kept in the `model_data` volume. To retrain:

```bash
docker compose run --rm -e FORCE_RETRAIN=1 trainer
docker compose run --rm -e FORCE_RETRAIN=1 -e EPOCHS=20 -e AUGMENT=0 trainer
docker compose restart webapp          # load the new weights
```

Training settings are read from environment variables, so changing them needs no rebuild:

| Variable | Default | Meaning |
|---|---|---|
| `EPOCHS` | 10 | training epochs |
| `BATCH_SIZE` | 256 | mini-batch size |
| `AUGMENT` | 1 | random rotation, scale and shift of training images, on the GPU |
| `FORCE_RETRAIN` | 0 | retrain even if a saved model exists |

## 3. Check that it really runs on the GPU

```bash
curl http://localhost:8000/health      # "device": "cuda:0", "gpu": "<your GPU>", ...
nvidia-smi                             # a python process appears while the containers run
```

If CUDA is not available inside a container, both services stop with an explicit error instead of silently falling back to the CPU (`REQUIRE_GPU=1`).

## Differences from mnist-app-cpu

| | cpu | gpu |
|---|---|---|
| PyTorch | CPU-only build | CUDA build, GPU reserved in `docker-compose.yml` |
| Docker images | two, each with its own PyTorch download | one, shared by both services |
| model class | copied in two files | `lenet.py`, imported by both |
| data loading | DataLoader, one PIL image at a time | whole MNIST in GPU memory, batches sliced on the GPU |
| augmentation | none | random rotation, scale and shift on the GPU |
| evaluation | none | test accuracy every epoch, saved to `lenet_mnist.json` |
| model hand-off | web app polls for the file and can read it half-written | atomic save, and the web app waits for the trainer to finish |
| retraining | on every `docker compose up` | only when asked (`FORCE_RETRAIN=1`) |
| MNIST download | on every training run | cached in the `mnist_data` volume |
| upload preprocessing | squash to 28 x 28, always invert | invert only if the background is light, crop to the ink, fit into 20 x 20, centre by mass, as in MNIST |

## What the GPU speeds up, and what it doesn't

Training is where the GPU pays off: with the dataset already in GPU memory, an epoch takes seconds.

For a single 28 x 28 image, the forward pass through LeNet takes a fraction of a millisecond on the GPU. The time of a request is dominated by HTTP, image decoding and preprocessing, so the web app does not feel faster than the CPU version. A GPU makes a difference for serving when requests are batched or the model is much larger, which is what later chapters of the course get to.
