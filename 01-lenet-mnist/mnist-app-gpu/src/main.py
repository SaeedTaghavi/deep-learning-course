"""
main.py -- FastAPI web app that classifies an uploaded digit image with LeNet on the GPU.

Motivation
----------
Serving counterpart of train.py: loads the weights the trainer wrote to the
shared volume onto the RTX 4070 and answers predictions over HTTP.

Two things are changed with respect to mnist-app-cpu, besides the device:
1. Preprocessing. The CPU app squashes the whole upload to 28x28 and always
   inverts it. MNIST digits are white on black, cropped to fit a 20x20 box and
   centred by centre of mass in a 28x28 frame; an upload that does not look like
   that is out of distribution and gets misclassified even by a 99%-accurate
   model. Here the image is inverted only if its background is bright,
   contrast-stretched, cropped to the ink, scaled into 20x20 and centred.
2. Visibility. Each response includes the 28x28 image the network actually saw,
   so a wrong prediction can be traced to preprocessing or to the model.

What it measures
----------------
Per /predict request: predicted digit, softmax confidence, top-3 classes, the
device used, and the model forward time in ms (CUDA-synchronised, so it is real
GPU time and not just kernel launch time). /health reports the GPU, the torch and
CUDA versions, and the test accuracy recorded by the trainer.

Usage
-----
    docker compose up --build                     # then open http://localhost:8000
    curl http://localhost:8000/health
    curl -F "file=@my_digit.png" http://localhost:8000/predict

Directly, next to a trained model:
    SHARED_DIR=./shared_data uvicorn main:app --port 8000
"""
import base64
import io
import json
import os
import time
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageOps, UnidentifiedImageError

from lenet import LeNet, describe_device, pick_device

SHARED_DIR = os.environ.get("SHARED_DIR", "/shared_data")
MODEL_PATH = os.path.join(SHARED_DIR, "lenet_mnist.pth")
META_PATH = os.path.join(SHARED_DIR, "lenet_mnist.json")
REQUIRE_GPU = os.environ.get("REQUIRE_GPU", "1").strip().lower() in ("1", "true", "yes")
MODEL_WAIT_SECONDS = int(os.environ.get("MODEL_WAIT_SECONDS", 600))

device = pick_device(REQUIRE_GPU)
model = LeNet().to(device)
state = {"device_info": describe_device(device), "model_meta": None}


# ---------------------------------------------------------------- preprocessing
def _shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate a 2D array with zero fill (np.roll would wrap ink around the edges)."""
    out = np.zeros_like(a)
    h, w = a.shape
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = a[ys, xs]
    return out


def to_mnist_format(img: Image.Image) -> np.ndarray:
    """Turn an arbitrary digit image into a 28x28 float array in [0,1] laid out like MNIST."""
    g = np.asarray(ImageOps.exif_transpose(img).convert("L"), dtype=np.float32) / 255.0

    # MNIST is light ink on a dark background: invert if the border is bright.
    border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    if border.mean() > 0.5:
        g = 1.0 - g

    # Stretch contrast so the background (the median pixel) goes to 0 and the ink to 1.
    lo, hi = float(np.median(g)), float(g.max())
    if hi - lo < 0.05:
        return np.zeros((28, 28), np.float32)  # blank image
    g = np.clip((g - lo) / (hi - lo), 0.0, 1.0)

    # Crop to the ink.
    mask = g > 0.25
    rows, cols = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
    crop = g[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    # Fit the longest side into 20 px, keep aspect ratio.
    h, w = crop.shape
    s = 20.0 / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    crop_img = Image.fromarray((crop * 255).astype(np.uint8)).resize((nw, nh), Image.LANCZOS)
    crop = np.asarray(crop_img, dtype=np.float32) / 255.0

    canvas = np.zeros((28, 28), np.float32)
    top, left = (28 - nh) // 2, (28 - nw) // 2
    canvas[top:top + nh, left:left + nw] = crop

    # Move the centre of mass to the middle of the frame, as in MNIST.
    m = canvas.sum()
    if m > 0:
        yy, xx = np.indices(canvas.shape)
        cy, cx = (yy * canvas).sum() / m, (xx * canvas).sum() / m
        canvas = _shift(canvas, int(round(13.5 - cy)), int(round(13.5 - cx)))
    return canvas


def png_b64(a: np.ndarray, upscale: int = 6) -> str:
    im = Image.fromarray((a * 255).astype(np.uint8)).resize(
        (28 * upscale, 28 * upscale), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- lifecycle
@asynccontextmanager
async def lifespan(_: FastAPI):
    # docker compose only starts this service after the trainer exits successfully,
    # but wait anyway so the app also works when started by hand.
    t0 = time.time()
    while not os.path.exists(MODEL_PATH):
        if time.time() - t0 > MODEL_WAIT_SECONDS:
            raise RuntimeError(f"No model at {MODEL_PATH} after {MODEL_WAIT_SECONDS} s")
        print(f"Waiting for {MODEL_PATH} ...", flush=True)
        time.sleep(2)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            state["model_meta"] = json.load(f)

    # Warm-up: the first CUDA call initialises cuDNN and takes ~1 s; pay it now,
    # not on the first user request.
    with torch.inference_mode():
        model(torch.zeros(1, 1, 28, 28, device=device))
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("Model loaded on", state["device_info"], flush=True)
    yield


app = FastAPI(title="MNIST predictor (GPU)", lifespan=lifespan)
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health")
def health():
    return {"status": "ok", **state["device_info"], "model": state["model_meta"]}


# Plain `def`: FastAPI runs it in a worker thread, so GPU work does not block the event loop.
@app.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(file.file.read()))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not read that file as an image.")

    arr = to_mnist_format(image)
    x = torch.from_numpy(arr)[None, None].to(device, non_blocking=True)

    with torch.inference_mode():
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        probs = model(x).exp()[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
    top_p, top_i = probs.topk(3)
    top_p, top_i = top_p.cpu().tolist(), top_i.cpu().tolist()
    all_p = probs.cpu().tolist()

    return {
        "prediction": top_i[0],
        "confidence": round(top_p[0], 4),
        "top3": [{"digit": d, "p": round(p, 4)} for d, p in zip(top_i, top_p)],
        "probabilities": [round(p, 4) for p in all_p],
        "device": state["device_info"].get("gpu", str(device)),
        "inference_ms": round(ms, 3),
        "model_input_png": png_b64(arr),
    }
