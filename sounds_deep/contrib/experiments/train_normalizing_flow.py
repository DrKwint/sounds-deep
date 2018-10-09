from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import operator
from functools import reduce

import numpy as np
import sonnet as snt
import tensorflow as tf
import scipy.misc
from random import shuffle

import sounds_deep.contrib.data.data as data
import sounds_deep.contrib.util.util as util

tfd = tf.contrib.distributions
tfb = tfd.bijectors
parser = argparse.ArgumentParser(description='Train a VAE model.')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=5e-6)
args = parser.parse_args()

# load the data
# train_data, _, _, _ = data.load_cifar10('./data/')
train_data, _, _, _ = data.load_mnist('./data/')
train_data = np.reshape(train_data, [-1, 28, 28, 1])
# train_data, _, _, _ = data.load_sudoku('./data')
train_data = train_data.astype(np.float32)
data_shape = (args.batch_size, ) + train_data.shape[1:]
batches_per_epoch = train_data.shape[0] // args.batch_size
train_gen = data.data_generator(train_data, args.batch_size)


def fnfn(i):
    def _fn(x, output_units):
        first = snt.Linear(512)
        net = snt.Sequential([
            first, tf.nn.relu,
            snt.Linear(512), tf.nn.relu,
            snt.Linear(
                output_units * 2,
                initializers={
                    'w': tf.initializers.zeros(),
                    'b': tf.initializers.zeros()
                }), lambda x: tf.split(x, 2, axis=-1)
        ])
        shift, log_scale = net(x)
        # log_scale = tf.Print(log_scale, [tf.reduce_mean(x), tf.reduce_mean(first._w), tf.reduce_sum(log_scale, axis=-1)], message="{}: ".format(i))
        return shift, log_scale

    return tf.make_template("real_nvp_default_template", _fn)


def flow_step():
    bijectors = []
    bijectors.append(tfb.BatchNormalization())
    bijectors.append(tfb.Permute(permutation=list(reversed(range(784)))))
    bijectors.append(
        tfb.RealNVP(num_masked=784 // 2, shift_and_log_scale_fn=fnfn(i)))
    return tfb.Chain(bijectors)


bijectors = []
num_bijectors = 32
for i in range(num_bijectors):
    bijectors.append(flow_step())
flow_bijector = tfb.Chain(bijectors)

model = tfd.TransformedDistribution(
    distribution=tfd.MultivariateNormalDiag(
        loc=tf.zeros([args.batch_size, 784]),
        scale_diag=tf.ones([args.batch_size, 784])),
    bijector=flow_bijector)
print(model)

# build model
data_ph = tf.placeholder(tf.float32, shape=[args.batch_size, 784])
log_likelihood = model.log_prob(data_ph)
sample = tf.reshape(model.sample(), [args.batch_size, 28, 28])
optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
train_op = optimizer.minimize(-log_likelihood)

verbose_ops_dict = dict()
verbose_ops_dict['nll'] = -log_likelihood

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as session:
    session.run(tf.global_variables_initializer())
    print(session.run(-log_likelihood, {data_ph: next(train_gen)}))
    for epoch in range(args.epochs):
        print("EPOCH {}".format(epoch))
        out_dict = util.run_epoch_ops(
            session,
            train_data.shape[0] // args.batch_size,
            verbose_ops_dict=verbose_ops_dict,
            silent_ops=[train_op],
            feed_dict_fn=lambda: {data_ph: next(train_gen)},
            verbose=True)

        mean_nll = np.mean(out_dict['nll'])

        bits_per_dim = mean_nll / (
            np.log(2.) * reduce(operator.mul, data_shape[1:]))
        print("bits per dim: {:7.5f}\tnll: {:7.5f}".format(
            bits_per_dim, mean_nll))

        generated_img = session.run(sample)
        print(np.min(generated_img))
        print(np.max(generated_img))
        for i in range(generated_img.shape[0]):
            scipy.misc.toimage(generated_img[i]).save('epoch{}_{}.jpg'.format(
                epoch, i))
        sample_val = session.run(sample)
