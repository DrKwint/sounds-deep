from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import operator
from functools import reduce

import numpy as np
import sonnet as snt
import tensorflow as tf

import sounds_deep.contrib.data.data as data
import sounds_deep.contrib.util as util
import sounds_deep.contrib.models.vae as vae
import sounds_deep.contrib.parameterized_distributions.softplus_logistic as softplus_logistic

parser = argparse.ArgumentParser(description='Train a VAE model.')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--latent_dimension', type=int, default=32)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--learning_rate', type=float, default=3e-4)
args = parser.parse_args()

# load the data
# train_data, _, _, _ = data.load_cifar10('./data/')
train_data, _, _, _ = data.load_mnist('./data/')
train_data = np.reshape(train_data, [-1, 28, 28, 1])
data_shape = (args.batch_size, ) + train_data.shape[1:]
batches_per_epoch = train_data.shape[0] // args.batch_size
train_gen = data.data_generator(train_data, args.batch_size)

# build the model
encoder_module = snt.Sequential([
    snt.nets.ConvNet2D([16, 32, 64], [3], [1], [snt.SAME]),
    tf.keras.layers.Flatten()
])
decoder_module = snt.Sequential([
    snt.Linear(49), lambda x: tf.reshape(x, [-1, 7, 7, 1]),
    snt.nets.ConvNet2DTranspose([32, 16, 1], [(14, 14), (28,28), (28,28)], [3], [2, 2, 1], [snt.SAME])
])
model = vae.VAE(
    args.latent_dimension,
    encoder_module,
    decoder_module,
    output_dist_fn=softplus_logistic.SoftplusLogisticDistribution())

# build model
data_ph = tf.placeholder(tf.float32, shape=data_shape)
model(data_ph)
sample = model.sample()
print(sample)
exit()

global_step = tf.train.get_or_create_global_step()
learning_rate = tf.train.cosine_decay(args.learning_rate, global_step,
                                      args.epochs * batches_per_epoch)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(-model.elbo)

verbose_ops_dict = dict()
verbose_ops_dict['elbo'] = model.elbo
verbose_ops_dict['iw_elbo'] = model.importance_weighted_elbo

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as session:
    session.run(tf.global_variables_initializer())
    for epoch in range(args.epochs):
        print("EPOCH {}".format(epoch))
        out_dict = util.run_epoch_ops(
            session,
            train_data.shape[0] // args.batch_size,
            verbose_ops_dict=verbose_ops_dict,
            silent_ops=[train_op],
            feed_dict_fn=lambda: {data_ph: next(train_gen)},
            verbose=True)

        mean_elbo = np.mean(out_dict['elbo'])
        mean_iw_elbo = np.mean(out_dict['iw_elbo'])

        bits_per_dim = -mean_elbo / (
            np.log(2.) * reduce(operator.mul, data_shape[1:]))
        print("bits per dim: {:7.5f}\telbo: {:7.5f}\tiw_elbo: {:7.5f}".format(
            bits_per_dim, mean_elbo, mean_iw_elbo))
