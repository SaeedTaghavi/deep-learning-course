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
