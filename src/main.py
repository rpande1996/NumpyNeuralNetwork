import gzip
import os
from urllib.request import urlretrieve
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

"""## Helper Functions:

### Code (10 pts)
"""


def read_mnist(path=None):
    r"""Return (train_images, train_labels, test_images, test_labels).

    Args:
        path (str): Directory containing MNIST. Default is
            /home/USER/data/mnist or C:\Users\USER\data\mnist.
            Create if nonexistant. Download any missing files.

    Returns:
        Tuple of (train_images, train_labels, test_images, test_labels), each
            a matrix. Rows are examples. Columns of images are pixel values.
            Columns of labels are a onehot encoding of the correct class.
    """
    url = 'http://yann.lecun.com/exdb/mnist/'
    files = ['train-images-idx3-ubyte.gz',
             'train-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz']

    if path is None:
        # Set path to /home/USER/data/mnist or C:\Users\USER\data\mnist
        path = os.path.join(os.path.expanduser('~'), 'data', 'mnist')

    # Create path if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # Download any missing files
    for file in files:
        if file not in os.listdir(path):
            urlretrieve(url + file, os.path.join(path, file))
            print("Downloaded %s to %s" % (file, path))

    def _images(path):
        """Return images loaded locally."""
        with gzip.open(path) as f:
            # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
            pixels = np.frombuffer(f.read(), 'B', offset=16)
        return pixels.reshape(-1, 784).astype('float32') / 255

    def _labels(path):
        """Return labels loaded locally."""
        with gzip.open(path) as f:
            # First 8 bytes are magic_number, n_labels
            integer_labels = np.frombuffer(f.read(), 'B', offset=8)

        def _onehot(integer_labels):
            """Return matrix whose rows are onehot encodings of integers."""
            n_rows = len(integer_labels)
            n_cols = integer_labels.max() + 1
            onehot = np.zeros((n_rows, n_cols), dtype='uint8')
            onehot[np.arange(n_rows), integer_labels] = 1
            return onehot

        return _onehot(integer_labels)

    train_images = _images(os.path.join(path, files[0]))
    train_labels = _labels(os.path.join(path, files[1]))
    test_images = _images(os.path.join(path, files[2]))
    test_labels = _labels(os.path.join(path, files[3]))

    return train_images, train_labels, test_images, test_labels


def init_weights(input, y, n_hidden, neurons, n_input=None, n_output=None):
    if n_input is not None:
        inp_val = n_input
    else:
        inp_val = input.shape[1]
    if n_output is not None:
        out_val = n_output
    else:
        out_val = y.shape[1]
    biases = []
    weights = []
    weights.append(np.random.randn(inp_val, neurons[0]) * np.sqrt(0.006))
    for i in range(n_hidden):
        weights.append(np.random.randn(weights[i].shape[1], neurons[i + 1]) * np.sqrt(0.006))

    weights.append(np.random.randn(weights[-1].shape[1], out_val) * np.sqrt(0.006))

    for i in range(len(weights)):
        biases.append(np.random.randn(weights[i].shape[1], ) * np.sqrt(0.006))

    return weights, biases


def ReLU(x):
    return np.maximum(0, x)


def dReLU(x):
    return 1 * (x > 0)


def softmax(z):
    z = z - np.max(z, axis=1).reshape(z.shape[0], 1)
    return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(z.shape[0], 1)


def shuffle(input, target):
    idx = [i for i in range(input.shape[0])]
    np.random.shuffle(idx)
    input = input[idx]
    target = target[idx]
    return input, target


def revl(sample_list):
    reversed_list = sample_list.copy()
    reversed_list.reverse()
    return reversed_list


def plot(val, ylab):
    plt.plot(val)
    plt.xlabel("Epochs")
    plt.ylabel(ylab)
    plt.title(ylab + ' vs Epochs')
    plt.show()


def feedforward(x, y, weights, biases):
    layer_output = []
    activation_layer = []

    assert x.shape[1] == weights[0].shape[0]
    layer_output.append(x.dot(weights[0]) + biases[0])
    activation_layer.append(ReLU(layer_output[0]))

    for i in range(1, len(weights)):
        assert activation_layer[i - 1].shape[1] == weights[i].shape[0]
        layer_output.append(activation_layer[i - 1].dot(weights[i]) + biases[i])
        activation_layer.append(ReLU(layer_output[i]))

    error = activation_layer[-1] - y
    return error, activation_layer, layer_output, weights, biases, x, y


def backprop(activation_layer, layer_output, weights, biases, batch, error, lr):
    dcost = (1 / batch) * error
    rev_activation = revl(activation_layer)
    rev_layer = revl(layer_output)
    rev_wts = revl(weights)
    rev_bias = revl(biases)
    vals = []
    b_vals = []
    del_wts = []
    del_bias = []
    del_wts.append(np.dot(dcost.T, rev_activation[1]).T)
    val = np.dot((dcost), rev_wts[0].T) * dReLU(rev_layer[1])
    vals.append(val)
    del_wts.append((np.dot(val.T, rev_activation[2])).T)

    for i in range(len(weights) - 3):
        val = np.dot(vals[i], rev_wts[i + 1].T) * dReLU(rev_layer[i + 2])
        vals.append(val)
        del_wts.append((np.dot(vals[i + 1].T, rev_activation[i + 3])).T)

    del_wts.append(np.dot((np.dot(vals[-1], rev_wts[-2].T) * dReLU(rev_layer[-1])).T, x).T)

    del_bias.append(np.sum(dcost, axis=0))

    b_val = np.dot((dcost), rev_wts[0].T) * dReLU(rev_layer[1])
    b_vals.append(b_val)
    del_bias.append(np.sum(b_val, axis=0))

    for i in range(len(weights) - 2):
        b_val = np.dot(b_vals[i], rev_wts[i + 1].T) * dReLU(rev_layer[i + 2])
        b_vals.append(b_val)
        del_bias.append(np.sum(b_vals[i + 1], axis=0))

    for i in range(len(weights)):
        assert del_wts[i].shape == rev_wts[i].shape
        rev_wts[i] = rev_wts[i] - lr * del_wts[i]
        assert del_bias[i].shape == rev_bias[i].shape
        rev_bias[i] = rev_bias[i] - lr * del_bias[i]

    rev_wts.reverse()
    rev_bias.reverse()

    return rev_wts, rev_bias


def train(input, target, weights, biases, batch, epochs, lr):
    loss = []
    acc = []
    for j in tqdm(range(epochs)):
        l = 0
        acc_val = 0
        input, target = shuffle(input, target)

        for i in range(input.shape[0] // batch - 1):
            start = i * batch
            end = (i + 1) * batch
            x = input[start:end]
            y = target[start:end]
            error, activation_layer, layer_output, weights, biases, x, y = feedforward(x, y, weights, biases)
            weights, biases = backprop(activation_layer, layer_output, weights, biases, batch, error, lr)
            l += np.mean(error ** 2)
            acc_val += np.count_nonzero(np.argmax(activation_layer[-1], axis=1) == np.argmax(y, axis=1)) / batch

        loss.append(l / (input.shape[0] // batch))
        acc.append(acc_val / (input.shape[0] // batch))
    print("Train Accuracy:", np.max(acc) * 100, "%")
    return weights, biases, loss, acc


def test(xtest, ytest, weights, biases):
    x = xtest
    y = ytest
    _, activation_layer, _, weights, biases, x, y = feedforward(x, y, weights, biases)
    acc_val = np.count_nonzero(np.argmax(activation_layer[-1], axis=1) == np.argmax(ytest, axis=1)) / xtest.shape[0]
    print("Test Accuracy:", 100 * acc_val, "%")


train_images, train_labels, test_images, test_labels = read_mnist(path='../input/data/')
input = train_images
target = train_labels
batch = 64
lr = 1e-3
epochs = 100
x = input[:batch]
y = target[:batch]

weights, biases = init_weights(input, y, 2, [256, 128, 50])
weights, biases, loss, acc = train(input, target, weights, biases, batch, epochs, lr)
test(test_images, test_labels, weights, biases)
plot(loss, 'Loss')
plot(acc, 'Accuracy')
