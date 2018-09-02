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
from sounds_deep.contrib.models.normalizing_flows import GlowFlow
from sounds_deep.contrib.models.normalizing_flows import NormalizingFlows
from sounds_deep.contrib.models.normalizing_flows import glow_net_fn

parser = argparse.ArgumentParser(description='Train a Glow model.')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=5e-6)
args = parser.parse_args()

# load the data
train_data, train_labels, _, _ = data.load_mnist('./data/')
train_data = np.reshape(train_data, [-1, 28, 28, 1])
data_shape = (args.batch_size, ) + train_data.shape[1:]
label_shape = (args.batch_size, ) + train_labels.shape[1:]
batches_per_epoch = train_data.shape[0] // args.batch_size
train_gen = data.parallel_data_generator([train_data, train_labels], args.batch_size)

def feed_dict_fn():
    feed_dict = dict()
    arrays = next(train_gen)
    feed_dict[data_ph] = arrays[0]
    feed_dict[label_ph] = arrays[1]
    return feed_dict

# build model
data_ph = tf.placeholder(tf.float32, shape=data_shape)
label_ph = tf.placeholder(tf.float32, shape=label_shape)

glow = GlowFlow(1, 16, glow_net_fn, flow_coupling_type='scale_and_shift')
model = NormalizingFlows(glow)

objective, stats_dict = model(data_ph, label_ph)

optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
train_op = optimizer.minimize(objective)

verbose_ops_dict = stats_dict
verbose_ops_dict['objective'] = objective

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
            feed_dict_fn=feed_dict_fn,
            verbose=True)

        mean_objective = np.mean(out_dict['objective'])
        print("objective: {:7.5f}".format(mean_objective))
