from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import operator
from functools import reduce

import numpy as np
import sonnet as snt
import tensorflow as tf
import scipy
import sklearn

import sounds_deep.contrib.data.data as data
import sounds_deep.contrib.util.scaling as scaling
import sounds_deep.contrib.util.util as util
import sounds_deep.contrib.models.cpvae as cpvae
import sounds_deep.contrib.models.vae as vae
import sounds_deep.contrib.parameterized_distributions.discretized_logistic as discretized_logistic

parser = argparse.ArgumentParser(description='Train a VAE model.')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--latent_dimension', type=int, default=64)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--learning_rate', type=float, default=3e-5)
parser.add_argument('--dataset', type=str, default='mnist')

parser.add_argument('--max_leaf_nodes', type=int, default=20)
parser.add_argument('--max_depth', type=int, default=10)

args = parser.parse_args()

# load the data
if args.dataset == 'cifar10':
    train_data, train_labels, _, _ = data.load_cifar10('./data/')
elif args.dataset == 'mnist':
    train_data, train_labels, _, _ = data.load_mnist('./data/')
    train_data = np.reshape(train_data, [-1, 28, 28, 1])
data_shape = (args.batch_size, ) + train_data.shape[1:]
label_shape = (args.batch_size, ) + train_labels.shape[1:]
train_batches_per_epoch = train_data.shape[0] // args.batch_size
train_gen = data.data_generator(train_data, args.batch_size)
train_gen = data.parallel_data_generator([train_data, train_labels],
                                         args.batch_size)

# build the model
if args.dataset == 'cifar10':
    encoder_module = snt.Sequential([
        snt.Conv2D(16, 3),
        snt.Residual(snt.Conv2D(16, 3)),
        snt.Residual(snt.Conv2D(16, 3)), scaling.squeeze2d,
        snt.Conv2D(64, 3),
        snt.Residual(snt.Conv2D(64, 3)),
        snt.Residual(snt.Conv2D(64, 3)), scaling.squeeze2d,
        snt.Conv2D(64, 3),
        snt.Residual(snt.Conv2D(64, 3)),
        snt.Residual(snt.Conv2D(64, 3)), scaling.squeeze2d,
        snt.Conv2D(128, 3),
        snt.Residual(snt.Conv2D(128, 3)),
        snt.Residual(snt.Conv2D(128, 3)), scaling.squeeze2d,
        snt.Conv2D(256, 3),
        snt.Residual(snt.Conv2D(256, 3)),
        snt.Residual(snt.Conv2D(256, 3)), scaling.squeeze2d,
        tf.keras.layers.Flatten(),
        snt.Linear(100)
    ])
    decoder_module = snt.Sequential([
        lambda x: tf.reshape(x, [-1, 4, 4, 4]),
        snt.Conv2D(32, 3),
        snt.Residual(snt.Conv2D(32, 3)),
        snt.Residual(snt.Conv2D(32, 3))
    ] + [
        scaling.unsqueeze2d,
        snt.Conv2D(32, 3),
        snt.Residual(snt.Conv2D(32, 3)),
        snt.Residual(snt.Conv2D(32, 3))
    ] * 5)
    output_distribution_fn = discretized_logistic.DiscretizedLogistic
elif args.dataset == 'mnist':
    encoder_module = snt.Sequential(
        [tf.keras.layers.Flatten(),
         snt.nets.MLP([200, 200])])
    decoder_module = snt.Sequential([
        lambda x: tf.reshape(x, [-1, 8, 8, 1]),
        snt.Residual(snt.Conv2D(1, 3)), lambda x: tf.reshape(x, [-1, 64]),
        snt.nets.MLP([200, 200, 784]),
        lambda x: tf.reshape(x, [-1, 28, 28, 1])
    ])
    output_distribution_fn = vae.BERNOULLI_FN

    def train_feed_dict_fn():
        feed_dict = dict()
        arrays = next(train_gen)
        feed_dict[data_ph] = arrays[0]
        feed_dict[label_ph] = arrays[1]
        return feed_dict


decision_tree = sklearn.tree.DecisionTreeClassifier(
    max_depth=args.max_depth, min_weight_fraction_leaf=0.01, max_leaf_nodes=args.max_leaf_nodes)
model = cpvae.CPVAE(
    args.latent_dimension,
    args.max_leaf_nodes,
    10,
    decision_tree,
    encoder_module,
    decoder_module,
    output_dist_fn=output_distribution_fn)

# build model
data_ph = tf.placeholder(
    tf.float32, shape=(args.batch_size, ) + data_shape[1:])
label_ph = tf.placeholder(
    tf.float32, shape=(args.batch_size, ) + label_shape[1:])
objective = model(data_ph, label_ph, analytic_kl=True)
sample = model.sample()

optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
train_op = optimizer.minimize(objective)

verbose_ops_dict = dict()
verbose_ops_dict['distortion'] = model.distortion
verbose_ops_dict['rate'] = model.rate
verbose_ops_dict['elbo'] = model.elbo
verbose_ops_dict['iw_elbo'] = model.importance_weighted_elbo
verbose_ops_dict['prior_logp'] = model.prior_logp
verbose_ops_dict['posterior_logp'] = model.posterior_logp
verbose_ops_dict['classification_loss'] = model.classification_loss

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as session:
    session.run(tf.global_variables_initializer())
    for epoch in range(args.epochs):
        print("EPOCH {}".format(epoch))
        train_class_rate = 1. - model.update(session, label_ph, 5*train_batches_per_epoch,
                     train_feed_dict_fn, epoch)

        out_dict = util.run_epoch_ops(
            session,
            train_data.shape[0] // args.batch_size,
            verbose_ops_dict=verbose_ops_dict,
            silent_ops=[train_op],
            feed_dict_fn=train_feed_dict_fn,
            verbose=True)

        mean_distortion = np.mean(out_dict['distortion'])
        mean_rate = np.mean(out_dict['rate'])
        mean_elbo = np.mean(out_dict['elbo'])
        mean_iw_elbo = np.mean(out_dict['iw_elbo'])
        mean_prior_logp = np.mean(out_dict['prior_logp'])
        mean_posterior_logp = np.mean(out_dict['posterior_logp'])
        mean_classification_loss = np.mean(out_dict['classification_loss'])

        bits_per_dim = -mean_elbo / (
            np.log(2.) * reduce(operator.mul, data_shape[-3:]))
        print(
            "bits per dim: {:7.5f}\tdistortion: {:7.5f}\trate: {:7.5f}\tprior_logp: \
            {:7.5f}\tposterior_logp: {:7.5f}\telbo: {:7.5f}\tiw_elbo: {:7.5f}\tclass_rate: {:7.5f}\tclass_loss: {:7.5f}"
            .format(bits_per_dim, mean_distortion, mean_rate, mean_prior_logp,
                    mean_posterior_logp, mean_elbo, mean_iw_elbo, train_class_rate, mean_classification_loss))

        generated_img = session.run(sample)
        for i in range(generated_img.shape[0]):
            img = np.clip(np.squeeze(generated_img[i]), 0, 1)
            scipy.misc.toimage(np.squeeze(generated_img[i])).save(
                'epoch{}_{}.jpg'.format(epoch, i))
