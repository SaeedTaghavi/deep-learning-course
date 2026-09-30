# mnist-app-cpu

LeNet-5 behind a web page, on the CPU. Upload a picture of a digit and the page shows the prediction, the probability of every digit, and the 28 x 28 image the network actually received.

It is built from two Docker containers that share a volume:

```
trainer  --- writes lenet_mnist.pth --->  [ model_data volume ]  --- reads --->  webapp (:8000)
```

```
mnist-app-cpu/
  docker-compose.yml       the two services and the shared volume
  model/
    Dockerfile.train       CPU PyTorch image for training
    train.py               trains 5 epochs and saves the weights to /shared_data
  app/
    Dockerfile.app         CPU PyTorch + FastAPI image
    main.py                loads the weights and serves / and /predict
    templates/index.html   the page
```

## Run

You need [Docker Desktop](https://www.docker.com/products/docker-desktop/) (on Windows, with the WSL 2 engine).

```bash
cd mnist-app-cpu
docker compose up --build
```

1. `trainer` downloads MNIST, trains for 5 epochs and prints the loss of each epoch, saves the weights and exits.
2. `webapp` waits until the weights file exists, loads it and serves the page at http://localhost:8000.

Stop with Ctrl+C, and remove the containers with `docker compose down`. Add `-v` to also delete the saved model.

## Using the page

Choose an image, drop it on the page, or paste it from the clipboard. The prediction starts straight away. Dark ink on a light background works best.

To get test images, extract some from MNIST with [`export_pngs.py`](../mnist-scripts-cpu#get-png-files-of-the-digits). Use `--invert`, because this app always inverts uploads and misreads MNIST's original white-on-black images:

```bash
cd ../mnist-scripts-cpu
python export_pngs.py --split test --per-digit 5 --invert --scale 10 --out mnist_pngs_upload
```

The same prediction is available from the command line:

```bash
curl -F "file=@my_digit.png" http://localhost:8000/predict
```

It returns the predicted digit, its confidence, the probabilities of all ten digits and the 28 x 28 input image as base64 PNG.

## Known limitations

These are left in on purpose, and [`mnist-app-gpu`](../mnist-app-gpu) shows how to fix each one:

- The model is retrained from scratch on every `docker compose up`.
- The `LeNet` class is copied in both `model/train.py` and `app/main.py`. If the two copies drift apart, the web app can't load the weights.
- The web app starts reading the weights file as soon as it appears, even if the trainer is still writing it.
- Uploads are squashed to 28 x 28 and always inverted, without cropping or centring the digit. A small digit in a big photo, or a white digit on black, is often misread. The "what the network sees" image on the page shows why.
- Both images download PyTorch separately.
- The model is never evaluated on the test set. See `test.py` in [`mnist-scripts-cpu`](../mnist-scripts-cpu).

The two apps both use port 8000, so run one at a time.
