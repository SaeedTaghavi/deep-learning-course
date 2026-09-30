# mnist-scripts-cpu

LeNet-5 on MNIST as two plain Python scripts on the CPU. No Docker, no web app, no GPU. Start here: the web apps in [`mnist-app-cpu`](../mnist-app-cpu) and [`mnist-app-gpu`](../mnist-app-gpu) wrap exactly this model and this training loop.

```
mnist-scripts-cpu/
  train.py            defines LeNet, trains it, saves lenet_mnist.pth
  test.py             evaluates the saved weights, or classifies your own image
  export_pngs.py      extracts the MNIST images as PNG files
  requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

The `--extra-index-url` gives the CPU-only build of PyTorch, about 200 MB instead of several GB for the CUDA build.

## Train

```bash
python train.py                      # 5 epochs, batch 64, Adam lr 1e-3
python train.py --epochs 10 --threads 8
```

Prints the loss, training accuracy and time for every epoch, then saves `lenet_mnist.pth`. MNIST is downloaded into `./data` on the first run.

## What's in the `data` folder

After the first run you will find this, and no image files:

```
data/MNIST/raw/
  train-images-idx3-ubyte     60,000 training images, all in one file
  train-labels-idx1-ubyte     their labels (0 to 9)
  t10k-images-idx3-ubyte      10,000 test images
  t10k-labels-idx1-ubyte      their labels
  *.gz                        the compressed files that were downloaded
```

This is the original format MNIST was published in, in 1998, called **IDX**. They are the same images you see in repositories such as [mnist-pngs](https://github.com/rasbt/mnist-pngs), which converted each one into a separate PNG so you can browse them. They are only stored differently:

- The file name describes the contents: `idx3` is a 3-dimensional array (image number x rows x columns), `idx1` is 1-dimensional (one label per image), and `ubyte` means every value is one unsigned byte, 0 to 255.
- An image file starts with a 16-byte header: a number identifying the file type (2051), the number of images, the number of rows (28) and the number of columns (28). After that come the pixels of every image, one byte each, image after image, row by row. Image number *i* is simply the 784 bytes that start at byte 16 + 784 x *i*.
- A label file has an 8-byte header, then one byte per label.

Why not PNGs? Reading one file in a single operation is much faster than opening 70,000 small ones, and it takes less space. torchvision reads these files straight into a single array of shape 60,000 x 28 x 28, which is what the training loop uses.

To see it for yourself, these lines read the format by hand and save the first training image as a PNG:

```python
import numpy as np
from PIL import Image

raw = "data/MNIST/raw/"
with open(raw + "train-images-idx3-ubyte", "rb") as f:
    magic, n, rows, cols = np.frombuffer(f.read(16), dtype=">u4")   # 4 big-endian integers
    images = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)
with open(raw + "train-labels-idx1-ubyte", "rb") as f:
    magic_l, n_l = np.frombuffer(f.read(8), dtype=">u4")
    labels = np.frombuffer(f.read(), dtype=np.uint8)

print(magic, n, rows, cols)            # 2051 60000 28 28
print(labels[:10])                     # the first ten digits
Image.fromarray(images[0]).resize((280, 280), Image.NEAREST).save("first_digit.png")
```

`>u4` means a 4-byte unsigned integer in big-endian byte order, the order used by the Sun workstations of the time, not the little-endian order of today's PCs.

To turn the files into ordinary images, use `export_pngs.py`, described [below](#get-png-files-of-the-digits).

**About the "HTTP Error 404" lines on the first run:** torchvision first tries the original download address at yann.lecun.com, which no longer serves the files, and then downloads them from a mirror. The messages are harmless. Once the files are in `./data`, they are not downloaded again.

## Test

```bash
python test.py                       # accuracy on the 10,000 test images
python test.py --save-errors errors.png
python test.py --image my_digit.png  # classify a picture of your own
```

On the test set it prints:

- overall accuracy
- accuracy for each digit
- the 10 x 10 confusion matrix (rows are the true digit, columns the prediction)
- the five most frequent mistakes

`--save-errors` also saves a grid of the misclassified images, which is the quickest way to see whether the mistakes are the network's fault or the handwriting's.

`--image` converts your picture to the MNIST layout first (light ink on dark, cropped, fitted into 20 x 20, centred) and prints the probability of every digit.

## Get PNG files of the digits

**Goal:** turn the raw MNIST files into ordinary image files. The downloaded files can't be opened in an image viewer (see [What's in the `data` folder](#whats-in-the-data-folder)), and the web apps only accept an image file as upload, so to look at the digits or test the apps with them you need PNGs first.

`export_pngs.py` writes the MNIST images out as PNG files, one folder per digit, so you can look at them or upload them to the web apps in [`mnist-app-cpu`](../mnist-app-cpu) and [`mnist-app-gpu`](../mnist-app-gpu). It only needs NumPy and Pillow, not PyTorch, and the raw files must already be in `./data` (run `train.py` or `test.py` once).

```bash
python export_pngs.py                                   # all 70,000 images -> ./mnist_pngs
python export_pngs.py --split test --per-digit 5        # 50 test images, 5 of each digit
```

```
mnist_pngs/
  train/0/00001.png  ...  train/9/...
  test/0/00003.png   ...  test/9/...
  labels.csv               split, index, label and path of every image
```

The number in each file name is the image's position in the original file.

**For uploading to the web apps**, make a small set of larger, dark-on-white images:

```bash
python export_pngs.py --split test --per-digit 5 --invert --scale 10 --out mnist_pngs_upload
```

- `--invert` saves a dark digit on a white background, like ink on paper. By default the PNGs keep MNIST's white digit on black. `mnist-app-cpu` always inverts uploads, so it misreads those; `mnist-app-gpu` detects the background and handles both.
- `--scale 10` enlarges the 28 x 28 images to 280 x 280 without blurring, so they are easier to see. The apps shrink them back to 28 x 28.

Use test images, not training images: the model has already seen every training image, so a correct answer on one of those tells you little.

## Keep each run in its own folder

To compare settings, give every run a folder with its weights and logs. `tee` shows the output on screen and saves it to a file at the same time:

```bash
mkdir -p runs/e10_t16
python -u train.py --epochs 10 --threads 16 --out runs/e10_t16/lenet_mnist.pth 2>&1 | tee runs/e10_t16/train.log
```

Then test that model and save its report in the same folder:

```bash
python -u test.py --weights runs/e10_t16/lenet_mnist.pth --save-errors runs/e10_t16/errors.png 2>&1 | tee runs/e10_t16/test.log
```

- Create the folder first with `mkdir -p`, because `tee` can't create it.
- `-u` makes Python write each line straight away, so the log fills in as training goes instead of in large chunks.
- `2>&1` saves warnings and errors to the log too.

The folder ends up with `lenet_mnist.pth`, `train.log`, `test.log` and `errors.png`.

## Things to try

- Train for 1 epoch, then for 10. How much does the test accuracy actually change?
- Compare training accuracy with test accuracy. Is the model overfitting?
- Change `--batch-size` to 8 and to 1024 and compare time per epoch and final accuracy.
- Write a digit on paper, photograph it and run `test.py --image`. Then try one written very thin, or very small in a corner of the photo.
- In `train.py`, remove the second convolution block (and change the input of `fc1` to match). What happens to the accuracy and the parameter count?
