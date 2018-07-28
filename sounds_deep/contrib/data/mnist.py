import urllib.request
import gzip
import pickle
import os
import numpy as np
from PIL import Image

img_size = 784

# Load the MNIST dataset
url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz'
}


def _download(data_dir, file_name):
    file_path = os.path.join(data_dir, file_name)

    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")


def download_mnist(data_dir):
    for v in key_file.values():
        _download(data_dir, v)


def _load_label(data_dir, file_name):
    file_path = os.path.join(data_dir, file_name)

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")

    return labels


def _load_img(data_dir, file_name):
    file_path = data_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("Done")

    return data


def _convert_numpy(data_dir):
    dataset = {}
    dataset['train_img'] = _load_img(data_dir, key_file['train_img'])
    dataset['train_label'] = _load_label(data_dir, key_file['train_label'])
    dataset['test_img'] = _load_img(data_dir, key_file['test_img'])
    dataset['test_label'] = _load_label(data_dir, key_file['test_label'])

    return dataset


def init_mnist(data_dir, save_file):
    download_mnist(data_dir)
    dataset = _convert_numpy(data_dir)
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done")


def _change_ont_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T


def load_mnist(data_dir, normalize=True, flatten=True, one_hot_label=True):
    """
    Parameters
    ----------
    normalize : Normalize the pixel values
    flatten : Flatten the images as one array
    one_hot_label : Encode the labels as a one-hot array

    Returns
    -------
    (Trainig Image, Training Label), (Test Image, Test Label)
    """
    save_file = os.path.join(data_dir, 'mnist.pkl')
    if not os.path.exists(save_file):
        init_mnist(data_dir, save_file)

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    if one_hot_label:
        dataset['train_label'] = _change_ont_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_ont_hot_label(dataset['test_label'])

    return (dataset['train_img'],
            dataset['train_label']), (dataset['test_img'],
                                      dataset['test_label'])


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
