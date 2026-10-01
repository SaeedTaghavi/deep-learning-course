# LeNet-5 on MNIST

The first chapter of the course: a small convolutional network that reads handwritten digits. The same model is built three times, from a bare training script to a web app trained and served on a GPU, so each step adds one new idea on top of code you have already read.

| Folder | What it is | Runs on | You need |
|---|---|---|---|
| [`mnist-scripts-cpu`](mnist-scripts-cpu) | `train.py` and `test.py`, nothing else | CPU | Python and PyTorch |
| [`mnist-app-cpu`](mnist-app-cpu) | the model behind a web page: upload a digit, get a prediction | CPU | Docker |
| [`mnist-app-gpu`](mnist-app-gpu) | the same web app, trained and served on an NVIDIA GPU | GPU | Docker and an NVIDIA GPU |

## Suggested order

1. **`mnist-scripts-cpu`**: the training loop, the loss, the optimizer, and how to evaluate a model honestly on data it never saw. Read `train.py` top to bottom, then run `test.py` and look at which digits it confuses.
2. **`mnist-app-cpu`**: how a trained model leaves the notebook. One container trains and writes the weights to a shared volume, a second container loads them and answers predictions over HTTP.
3. **`mnist-app-gpu`**: what changes when there is a GPU. CUDA builds of PyTorch, giving a container access to the GPU, keeping the whole dataset in GPU memory, data augmentation, and measuring what the GPU actually speeds up.

## The model

LeNet-5 (LeCun, Bottou, Bengio and Haffner, 1998) was one of the first convolutional networks used in practice, for reading handwritten digits on bank cheques. The version here uses ReLU and max-pooling instead of the original tanh and average pooling.

```
input            1 x 28 x 28     grayscale digit
conv 5x5, 20    20 x 24 x 24     -> ReLU
max-pool 2x2    20 x 12 x 12
conv 5x5, 50    50 x  8 x  8     -> ReLU
max-pool 2x2    50 x  4 x  4     flattened to 800
linear          500              -> ReLU
linear          10               -> log-softmax, one score per digit
```

About 431k parameters, most of them (400k) in the first fully connected layer. It reaches about 99% accuracy on the MNIST test set after a few epochs.

## The dataset

[MNIST](https://en.wikipedia.org/wiki/MNIST_database): 60,000 training and 10,000 test images of handwritten digits, 28 x 28 pixels, white ink on a black background, each digit scaled into a 20 x 20 box and centred by its centre of mass. That last detail matters when you test the model on your own handwriting: an image that is not laid out the same way is out of distribution. All three folders download MNIST automatically on the first run. The downloaded files are not images you can open, but the dataset's original binary format. [What's in the `data` folder](mnist-scripts-cpu/README.md#whats-in-the-data-folder) explains that format and shows how to read it by hand.

## Related datasets for further examples

MNIST has several close relatives with the same image size and file format. They make good next exercises, because the code in this chapter works on them with very few changes: usually only the dataset class and the number of outputs of the last layer. All of them are built into torchvision.

### EMNIST: handwritten letters and digits

[EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset) (Extended MNIST, Cohen et al., 2017) comes from the same NIST handwriting collection as MNIST and is processed the same way: 28 x 28 grayscale, white on black, IDX files. It is split in several ways:

| Split | Classes | Images | Contents |
|---|---|---|---|
| `letters` | 26 | 145,600 | the letters a to z, upper and lower case merged into one class per letter |
| `balanced` | 47 | 131,600 | digits and letters, the same number of images in every class |
| `byclass` | 62 | 814,255 | 0-9, A-Z and a-z as separate classes; very unbalanced |
| `bymerge` | 47 | 814,255 | like `byclass`, with look-alike upper and lower case pairs merged (c/C, o/O, s/S, ...) |
| `digits` | 10 | 280,000 | digits only, a larger alternative to MNIST |
| `mnist` | 10 | 70,000 | the same size and balance as MNIST |

```python
from torchvision.datasets import EMNIST
train = EMNIST(root="./data", split="letters", train=True, download=True)
```

Things to know before using it:

- The images are stored transposed. Displayed as they are, letters appear mirrored and rotated by 90 degrees. Transpose each image (`img.T`) before showing it or comparing it with your own uploads.
- In the `letters` split the labels run from 1 to 26, not from 0. Subtract 1, or give the network 27 outputs.
- Some classes can't be told apart even by a person: in handwriting, o/O, s/S or x/X look the same, and so can l/1/I. Accuracy is therefore well below MNIST's 99%, and the confusion matrix shows why. That makes it a good lesson in the limits set by the data rather than by the model.

### Fashion-MNIST: clothing

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) (Zalando Research, 2017) was designed as a drop-in replacement for MNIST: the same 60,000 training and 10,000 test images, 28 x 28 grayscale, 10 classes, same file format. The images are photos of clothing items instead of digits:

T-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot.

```python
from torchvision.datasets import FashionMNIST
train = FashionMNIST(root="./data", train=True, download=True)
```

It was made because MNIST had become too easy: almost any model scores above 98% on it, so it no longer tells good models from bad ones. On Fashion-MNIST the same LeNet reaches only around 90%, and most of its mistakes are between shirt, T-shirt, pullover and coat. Swapping it in is the quickest way to see how much harder a dataset can be with exactly the same shape.

### KMNIST: Japanese cursive characters

[KMNIST](https://github.com/rois-codh/kmnist) (Kuzushiji-MNIST, 2018) has 10 classes of cursive Japanese characters, written in a style that most modern readers can no longer read, in exactly MNIST's format and size. It is a test of whether a model works on a script that nobody tuned it for.

```python
from torchvision.datasets import KMNIST
train = KMNIST(root="./data", train=True, download=True)
```
