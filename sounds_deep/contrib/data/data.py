"""
Imports for each dataset are in the respective function so downstream consumers
don't have to install the dependencies for all of them.
"""
import os

import numpy as np
from tqdm import tqdm

import sounds_deep.contrib.data.util as util


def data_generator(array, batch_size):
    def inf_train_gen():
        while True:
            np.random.shuffle(array)
            for i in range(0, len(array) - batch_size + 1, batch_size):
                yield np.array(array[i:i + batch_size])

    return inf_train_gen()


def idxable_data_generator(idxable, idxs, batch_size):
    def inf_train_gen():
        while True:
            np.random.shuffle(idxs)
            for i in range(0, len(idxs) - batch_size + 1, batch_size):
                yield np.array([idxable[a] for a in idxs[i:i + batch_size]])

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
                yield [np.array(a[i:i + batch_size]) for a in arrays]

    return inf_train_gen(arrays)


def load_mnist(data_dir):
    import sounds_deep.contrib.data.mnist as mnist
    (train_data, train_labels), (test_data, test_labels) = mnist.load_mnist(
        data_dir, normalize=True, flatten=True, one_hot_label=True)
    return train_data, train_labels, test_data, test_labels


def load_fmnist(data_dir):
    import sounds_deep.contrib.data.fmnist as fmnist
    (train_data, train_labels), (test_data, test_labels) = fmnist.load_fmnist(
        data_dir, normalize=True, flatten=True, one_hot_label=True)
    return train_data, train_labels, test_data, test_labels


def load_cifar10(data_dir):
    import sounds_deep.contrib.data.cifar10 as cifar10
    """Returns CIFAR10 as (train_data, train_labels, test_data, test_labels
    
    Shapes are (50000, 32, 32, 3), (50000, 10), (10000, 32, 32, 3), (10000, 10)
    Data is in [0,1] and labels are one-hot
    """
    if not os.path.exists(os.path.join(data_dir, 'cifar10_train_data.npy')):
        cifar10.download_and_extract_npy(data_dir)

    train_data = np.load(os.path.join(
        data_dir, 'cifar10_train_data.npy')).astype('float32')
    train_labels = np.load(os.path.join(data_dir, 'cifar10_train_labels.npy'))
    train_labels = util.one_hot(train_labels, 10)
    test_data = np.load(os.path.join(
        data_dir, 'cifar10_test_data.npy')).astype('float32')
    test_labels = np.load(os.path.join(data_dir, 'cifar10_test_labels.npy'))
    test_labels = util.one_hot(test_labels, 10)
    return train_data, train_labels, test_data, test_labels


# def load_celeba(data_dir):
#     """
#     Returns:
#       4-tuple of an indexable of images, train indices, test indices, and attributes
#     """
#     import sounds_deep.contrib.data.celeba as celeba
#     idxable = celeba.CelebA(data_dir)
#     # train_idxs, val_idxs, test_idxs, attribute_names, attributes
#     return idxable, idxable.train_idxs, idxable.test_idxs, idxable.attributes

def load_celeba(data_dir):
    """Returns CelebA as (train_data, train_labels, test_data, test_labels)

        Shapes are (162770, 64, 64, 3), (162770, 2), (19962, 64, 64, 3), (19962, 10)
        Data is in [0,1] and labels are one-hot
    """
    train_data = np.load(os.path.join(data_dir, 'celeba_train_imgs.npy')) / 255.0
    test_data = np.load(os.path.join(data_dir, 'celeba_test_imgs.npy')) / 255.0

    info_pak = np.load(os.path.join(data_dir, 'celeba_attr.npz'))
    train_idxs = info_pak['train_idxs']
    val_idxs = info_pak['val_idxs']
    test_idxs = info_pak['test_idxs']
    attribute_names = info_pak['attribute_names']
    attributes = info_pak['attributes']
    male_attr_idx = 20

    def get_label(data, idxs):
        def jj(attr):
            important_attributes_idx = [0, 1, 4, 9, 16, 18, 22, 24, 29, 30, 34, 36, 37, 38]
            x = np.array([0 for i in range(attr.shape[0])])
            for i in important_attributes_idx:
                x = x + attr[:, i]
            return x

        label = attributes[idxs]
        sig = jj(label) >= 1
        label = label[sig]
        data = data[sig]

        label = label[:, 20].reshape([-1, 1])
        label = np.append(label, 1 - label, 1)
        return data, label

    train_data, train_label = get_label(train_data, train_idxs)
    test_data, test_label = get_label(test_data, test_idxs)

    return train_data, train_label, test_data, test_label


def load_sudoku(data_dir):
    train_data = np.load(os.path.join(data_dir, 'sudoku.npy'))
    train_labels = np.load(os.path.join(data_dir, 'sudoku_labels.npy'))
    return train_data, train_labels, None, None
