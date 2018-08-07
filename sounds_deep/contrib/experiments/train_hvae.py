from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import operator
from functools import reduce

import sonnet as snt
import tensorflow as tf
import numpy as np
import scipy

import sounds_deep.contrib.data.data as data
import sounds_deep.contrib.models.hvae as hvae
import sounds_deep.contrib.util as util

parser = argparse.ArgumentParser(description='Train a VAE model.')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--latent_dimension', type=int, default=32)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=0.001)
args = parser.parse_args()

def apply_temp(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = tf.log(a) / temperature
    a = tf.exp(a) / tf.reduce_sum(tf.exp(a), axis=1, keepdims=True)
    return a

# celebA data
# idxable, train_idxs, test_idxs, attributes = data.load_celeba('./sounds_deep/contrib/data/')
# batches_per_epoch = train_idxs.shape[0] // args.batch_size
# train_gen = data.idxable_data_generator(idxable, train_idxs, args.batch_size)
# data_shape = next(train_gen).shape

train_data, train_labels, _, _ = data.load_mnist('./data/')
data_shape = (args.batch_size, ) + train_data.shape[1:]
label_shape = (args.batch_size, ) + train_labels.shape[1:]
batches_per_epoch = train_data.shape[0] // args.batch_size
train_gen = data.parallel_data_generator([train_data, train_labels], args.batch_size)

# build the model
temp_ph = tf.placeholder(tf.float32)
encoder_module = snt.nets.MLP([200, 200])
decoder_module = snt.nets.MLP([200, 200, 784])
model = hvae.HVAE(args.latent_dimension, encoder_module, decoder_module, hvar_shape=10, temperature=temp_ph)

# build model
data_ph = tf.placeholder(tf.float32, shape=data_shape)
label_ph = tf.placeholder(tf.float32, shape=label_shape)
# label = apply_temp(tf.nn.softmax(label_ph), temperature=0.1)
# label = tf.Print(label, [label], summarize=10)
model(data_ph, label_ph, n_samples=50)
sample_img = tf.reshape(model.sample(), [-1, 28, 28])

optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(-model.elbo)

verbose_ops_dict = dict()
verbose_ops_dict['elbo'] = model.elbo
verbose_ops_dict['iw_elbo'] = model.importance_weighted_elbo

verbose_ops_dict['distortion'] = model.distortion
verbose_ops_dict['rate'] = model.rate
verbose_ops_dict['hrate'] = model.hrate
verbose_ops_dict['hvars_label'] = model.hvar_labels
verbose_ops_dict['hvar_cross_entropy'] = model.hvar_cross_entropy
verbose_ops_dict['hvars_sample'] = model.hvar_sample

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as session:
    session.run(tf.global_variables_initializer())
    temperature = .7
    for epoch in range(args.epochs):
        def feed_dict_fn():
            feed_dict = dict()
            arrays = next(train_gen)
            feed_dict[data_ph] = arrays[0]
            feed_dict[label_ph] = arrays[1]
            feed_dict[temp_ph] = temperature # float(1. / (epoch + 1))
            return feed_dict

        print("EPOCH {}".format(epoch))
        out_dict = util.run_epoch_ops(
            session,
            batches_per_epoch,
            verbose_ops_dict=verbose_ops_dict,
            silent_ops=[train_op],
            feed_dict_fn=feed_dict_fn,
            verbose=True)

        mean_elbo = np.mean(out_dict['elbo'])
        mean_iw_elbo = np.mean(out_dict['iw_elbo'])
        print(np.mean(out_dict['distortion']))
        print(np.mean(out_dict['rate']))
        print(np.mean(out_dict['hrate']))
        print(np.mean(out_dict['hvar_cross_entropy']))
        print(out_dict['hvars_sample'][0][0])
        print(out_dict['hvars_label'][0])

        bits_per_dim = -mean_elbo / (
            np.log(2.) * reduce(operator.mul, data_shape[1:]))
        print("bits per dim: {:7.5f}\telbo: {:7.5f}\tiw_elbo: {:7.5f}".format(
            bits_per_dim, mean_elbo, mean_iw_elbo))

        generated_img = session.run(sample_img, feed_dict={temp_ph: temperature})
        for i in range(generated_img.shape[0]):
            scipy.misc.toimage(generated_img[i]).save('epoch{}_{}.jpg'.format(epoch, i))