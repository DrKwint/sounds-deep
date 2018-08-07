from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import sounds_deep.contrib.data.data as data

parser = argparse.ArgumentParser(description='Train a VAE model.')
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--latent_dimension', type=int, default=50)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=0.001)
args = parser.parse_args()

# train_data, _, _, _ = data.load_mnist('./data/')
idxable, train_idxs, test_idxs, attributes = data.load_celeba('./data/')
batches_per_epoch = train_idxs.shape[0] // args.batch_size
train_gen = data.idxable_data_generator(idxable, train_idxs, args.batch_size)
data_shape = next(train_gen).shape
