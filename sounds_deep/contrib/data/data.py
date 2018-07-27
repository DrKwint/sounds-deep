import os

import numpy as np
from tqdm import tqdm

import sounds_deep.contrib.data.mnist as mnist
import sounds_deep.contrib.data.celeba as celeba
import sounds_deep.contrib.data.cifar10 as cifar10
import sounds_deep.contrib.data.util as util

def data_generator(array, batch_size):
    def inf_train_gen():
        while True:
            np.random.shuffle(array)
            for i in range(0, len(array) - batch_size + 1, batch_size):
                yield np.array(array[i:i+batch_size])
    return inf_train_gen()

def idxable_data_generator(idxable, idxs, batch_size):
    def inf_train_gen():
        while True:
            np.random.shuffle(idxs)
            for i in range(0, len(idxs) - batch_size + 1, batch_size):
                yield np.array([idxable[a] for a in idxs[i:i+batch_size]])
    return inf_train_gen()

def parallel_data_generator(arrays, batch_size):
    if not hasattr(arrays, '__iter__'): arrays = [arrays]
    def unison_shuffled_copies(arrays):
        assert all([len(a) == len(arrays[0]) for a in arrays])
        p = np.random.permutation(len(arrays[0]))
        return [a[p] for a in arrays]
    def inf_train_gen(arrays):
        while True:
            arrays = unison_shuffled_copies(arrays)
            for i in range(0, len(arrays[0]) - batch_size + 1, batch_size):
                yield [np.array(a[i:i+batch_size]) for a in arrays]
    return inf_train_gen(arrays)

def load_mnist(data_dir):
    (train_data, train_labels), (test_data, test_labels) = mnist.load_mnist(data_dir, normalize=True, flatten=True, one_hot_label=True)
    return train_data, train_labels, test_data, test_labels

def load_cifar10(data_dir):
    """Returns CIFAR10 as (train_data, train_labels, test_data, test_labels
    
    Shapes are (50000, 32, 32, 3), (50000, 10), (10000, 32, 32, 3), (10000, 10)
    Data is in [0,1] and labels are one-hot
    """
    if not os.path.exists(os.path.join(data_dir, 'cifar10_train_data.npy')):
        cifar10.download_and_extract_npy(data_dir)

    train_data = np.load(os.path.join(data_dir, 'cifar10_train_data.npy')).astype(
        'float32') / 255.0
    train_labels = np.load(os.path.join(data_dir, 'cifar10_train_labels.npy'))
    train_labels = util.one_hot(train_labels, 10)
    test_data = np.load(os.path.join(data_dir, 'cifar10_test_data.npy')).astype(
        'float32') / 255.0
    test_labels = np.load(os.path.join(data_dir, 'cifar10_test_labels.npy'))
    test_labels = util.one_hot(test_labels, 10)
    return train_data, train_labels, test_data, test_labels

def load_celeba(data_dir):
    idxable = celeba.CelebA(data_dir)
    # train_idxs, val_idxs, test_idxs, attribute_names, attributes
    return idxable, idxable.train_idxs, idxable.test_idxs, idxable.attributes

def load_sudoku(data_dir):
    train_data = np.load(os.path.join(data_dir, 'sudoku.npy'))
    train_labels = np.load(os.path.join(data_dir, 'sudoku_labels.npy'))
    return train_data, train_labels, None, None