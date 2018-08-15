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

import sounds_deep.contrib.data.data as data
import sounds_deep.contrib.util.scaling as scaling
import sounds_deep.contrib.util.util as util
import sounds_deep.contrib.models.vae as vae
import sounds_deep.contrib.models.nvvae as nvvae
import sounds_deep.contrib.parameterized_distributions.discretized_logistic as discretized_logistic

parser = argparse.ArgumentParser(description='Train a VAE model.')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--latent_dimension', type=int, default=32)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--learning_rate', type=float, default=3e-5)
parser.add_argument('--dataset', type=str, default='mnist')
args = parser.parse_args()


def apply_temp(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = tf.log(a) / temperature
    a = tf.exp(a) / tf.reduce_sum(tf.exp(a), axis=1, keepdims=True)
    return a


# load the data
if args.dataset == 'cifar10':
    train_data, train_labels, _, _ = data.load_cifar10('./data/')
elif args.dataset == 'mnist':
    train_data, train_labels, _, _ = data.load_mnist('./data/')
    train_data = np.reshape(train_data, [-1, 28, 28, 1])
# train_data = train_data[:5*args.batch_size]
# train_labels = train_labels[:5*args.batch_size]
data_shape = (args.batch_size, ) + train_data.shape[1:]
label_shape = (args.batch_size, ) + train_labels.shape[1:]
batches_per_epoch = train_data.shape[0] // args.batch_size
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
    nv_encoder_module = snt.nets.ConvNet2D([32, 64, 128], [5], [2], [snt.SAME])
    encoder_module = snt.nets.ConvNet2D([32, 64, 128], [5], [2], [snt.SAME])
    decoder_module = snt.Sequential([
        snt.Linear(64),
        lambda x: tf.reshape(x, [-1, 2, 2, 16]),
        snt.nets.ConvNet2DTranspose([128, 64, 32, 32], [(4,4),(8, 8), (16, 16),
                                                        (32, 32)], [5], [2],
                                    [snt.SAME]),
        lambda x: tf.reshape(x, [-1, 32*32*32]),
        snt.Linear(28*28),
        lambda x: tf.reshape(x, [-1, 28, 28, 1])
    ])
    output_distribution_fn = vae.BERNOULLI_FN

model = nvvae.NamedLatentVAE(
    args.latent_dimension,
    10,
    nv_encoder_module,
    encoder_module,
    decoder_module,
    output_dist_fn=output_distribution_fn)

# build model
temperature_ph = tf.placeholder(tf.float32)
data_ph = tf.placeholder(
    tf.float32, shape=(args.batch_size, ) + data_shape[1:])
label_ph = tf.placeholder(tf.float32, shape=label_shape)
model(data_ph, label_ph, temperature_ph, analytic_kl=True)

num_samples = 10
nv_sample_ph = tf.placeholder_with_default(tf.ones([num_samples,10]), [num_samples, 10])
sample = model.sample(sample_shape=[num_samples], temperature=temperature_ph, nv_prior_sample=nv_sample_ph)

optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
train_op = optimizer.minimize(-model.elbo)

verbose_ops_dict = dict()
verbose_ops_dict['distortion'] = model.distortion
verbose_ops_dict['rate'] = model.rate
verbose_ops_dict['nv_rate'] = model.nv_rate
verbose_ops_dict['elbo'] = model.elbo
verbose_ops_dict['iw_elbo'] = model.importance_weighted_elbo
verbose_ops_dict['prior_logp'] = model.prior_logp
verbose_ops_dict['posterior_logp'] = model.posterior_logp
verbose_ops_dict['nv_prior_logp'] = model.nv_prior_logp
verbose_ops_dict['nv_posterior_logp'] = model.nv_posterior_logp

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as session:
    session.run(tf.global_variables_initializer())
    for epoch in range(args.epochs):
        # temperature = 0.3
        temperature = 2./3.
        # temperature = np.max(
        #     [0.5, np.exp(1e-5 * -float(epoch * train_data.shape[0]))])
        # temperature = float(10. / (epoch + 1))
        print("Temperature: {}".format(temperature))

        def feed_dict_fn():
            feed_dict = dict()
            arrays = next(train_gen)
            feed_dict[data_ph] = arrays[0]
            feed_dict[label_ph] = arrays[1]
            feed_dict[temperature_ph] = temperature
            return feed_dict

        print("EPOCH {}".format(epoch))
        out_dict = util.run_epoch_ops(
            session,
            train_data.shape[0] // args.batch_size,
            verbose_ops_dict=verbose_ops_dict,
            silent_ops=[train_op],
            feed_dict_fn=feed_dict_fn,
            verbose=True)

        mean_distortion = np.mean(out_dict['distortion'])
        mean_rate = np.mean(out_dict['rate'])
        mean_nv_rate = np.mean(out_dict['nv_rate'])
        mean_elbo = np.mean(out_dict['elbo'])
        mean_iw_elbo = np.mean(out_dict['iw_elbo'])
        mean_prior_logp = np.mean(out_dict['prior_logp'])
        mean_posterior_logp = np.mean(out_dict['posterior_logp'])
        mean_nv_prior_logp = np.mean(out_dict['nv_prior_logp'])
        mean_nv_posterior_logp = np.mean(out_dict['nv_posterior_logp'])

        bits_per_dim = -mean_elbo / (
            np.log(2.) * reduce(operator.mul, data_shape[-3:]))
        print(
            "bits per dim: {:7.4f}\tdistortion: {:7.4f}\trate: {:7.4f}\tnv_rate: {:7.4f}\tprior_logp: {:7.4f}\tposterior_logp: {:7.4f}\telbo: {:7.4f}\tiw_elbo: {:7.4f}"
            .format(bits_per_dim, mean_distortion, mean_rate, mean_nv_rate,
                    mean_prior_logp, mean_posterior_logp, mean_elbo,
                    mean_iw_elbo))

        for c in range(10):
            nv_sample_val = np.zeros([num_samples, 10], dtype=float)
            nv_sample_val[c] = 1.
            generated_img = session.run(sample, {temperature_ph: temperature, nv_sample_ph: nv_sample_val})
            for i in range(generated_img.shape[0]):
                img = np.clip(np.squeeze(generated_img[i]), 0, 1)
                scipy.misc.toimage(np.squeeze(generated_img[i])).save(
                    'epoch{}_class{}_{}.jpg'.format(epoch, c, i))
