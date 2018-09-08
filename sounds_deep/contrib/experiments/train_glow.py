"""Train a Glow normalizing flows model. TODO: fix stats dicts from towers """
from __future__ import absolute_import, division, print_function

import argparse
import operator
import os
from functools import reduce
from random import shuffle

import numpy as np
import scipy.misc
import sonnet as snt
import tensorflow as tf

import sounds_deep.contrib.data.data as data
import sounds_deep.contrib.util.plot as plot
import sounds_deep.contrib.util.util as util
from sounds_deep.contrib.models.normalizing_flows import (GlowFlow,
                                                          NormalizingFlows,
                                                          glow_net_fn)

parser = argparse.ArgumentParser(description='Train a Glow model.')
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--levels', type=int, default=2)
parser.add_argument('--depth_per_level', type=int, default=16)
parser.add_argument('--output_dir', type=str, default='')
parser.add_argument('--num_gpus', type=int, default=1)
# logscale factor for actnorm: 0.1 works well, must be <3
args = parser.parse_args()

if args.num_gpus > 0:
    device_string = 'gpu'
else:
    device_string = 'cpu'
    args.num_gpus = 1

# sampled img save directory
if args.output_dir == '' and 'SLURM_JOB_ID' in os.environ.keys():
    job_id = os.environ['SLURM_JOB_ID']
    output_directory = 'glow_{}'.format(job_id)
    os.mkdir(output_directory)
else:
    if args.output_dir == '':
        args.output_dir = './'
    else:
        output_directory = args.output_dir
        os.mkdir(output_directory)

# load the data
if args.dataset == 'cifar10':
    train_data, train_labels, _, _ = data.load_cifar10('./data/')
    train_data *= 255
elif args.dataset == 'mnist':
    train_data, train_labels, test_data, test_labels = data.load_mnist(
        './data/')
    train_data *= 255
    train_data = np.reshape(train_data, [-1, 28, 28, 1])
data_shape = (args.batch_size, ) + train_data.shape[1:]
label_shape = (args.batch_size, ) + train_labels.shape[1:]
batches_per_epoch = train_data.shape[0] // (args.batch_size * args.num_gpus)
train_gen = data.parallel_data_generator([train_data, train_labels],
                                         args.batch_size)

# build model
glow = GlowFlow(
    args.levels,
    args.depth_per_level,
    glow_net_fn,
    flow_coupling_type='scale_and_shift')
model = NormalizingFlows(glow)
optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)

data_ph_list = []
label_ph_list = []
objective_list = []
stats_dict_list = []
with tf.variable_scope(tf.get_variable_scope()):
    for i in range(args.num_gpus):
        with tf.device('/%s:%d' % (device_string, i)):
            with tf.name_scope('tower_%d' % i) as scope:
                data_ph = tf.placeholder(tf.float32, shape=data_shape)
                label_ph = tf.placeholder(tf.float32, shape=label_shape)
                objective, stats_dict = model(data_ph, label_ph)
                data_ph_list.append(data_ph)
                label_ph_list.append(label_ph)
                objective_list.append(objective)
                stats_dict_list.append(stats_dict)

average_grads = util.average_gradients([optimizer.compute_gradients(obj) for obj in objective_list])
train_op = optimizer.apply_gradients(average_grads)

def feed_dict_fn():
    feed_dict = dict()
    for data_ph, label_ph in zip(data_ph_list, label_ph_list):
        arrays = next(train_gen)
        feed_dict[data_ph] = arrays[0]
        feed_dict[label_ph] = arrays[1]
    return feed_dict

# setup sampling
sample_label_ph = tf.placeholder(tf.int32, shape=(16))
sample = model.sample(tf.one_hot(sample_label_ph, 10))

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
            batches_per_epoch,
            verbose_ops_dict=verbose_ops_dict,
            silent_ops=[train_op],
            feed_dict_fn=feed_dict_fn,
            verbose=True)

        mean_objective = np.mean(out_dict['objective'])
        print("objective: {:7.5f}".format(mean_objective))

        for i in range(10):
            sample_val = session.run(
                sample, feed_dict={sample_label_ph: np.ones(16) * i})
            filename = os.path.join(output_directory,
                                    'epoch{}_class{}.png'.format(epoch, i))
            plot.plot(filename, np.squeeze(sample_val), 4, 4)
