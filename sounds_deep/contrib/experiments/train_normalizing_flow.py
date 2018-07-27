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

import sounds_deep.contrib.data.data as data
import util

tfd = tf.contrib.distributions
tfb = tfd.bijectors
parser = argparse.ArgumentParser(description='Train a VAE model.')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--latent_dimension', type=int, default=50)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=3e-4)
args = parser.parse_args()

# load the data
# train_data, _, _, _ = data.load_cifar10('./data/')
# train_data, _, _, _ = data.load_mnist('./data/')
train_data, _, _, _ = data.load_sudoku('./data')
train_data = np.reshape(train_data, [-1, 81, 9])
train_data = train_data.astype(np.float32)
train_data += 5e-2
data_shape = (args.batch_size, ) + train_data.shape[1:]
batches_per_epoch = train_data.shape[0] // args.batch_size
train_gen = data.data_generator(train_data, args.batch_size)

bijectors = []
num_bijectors = 1
for i in range(num_bijectors):
    bijectors.append(
        tfb.MaskedAutoregressiveFlow(
            shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
                hidden_layers=[512, 512])))
    bijectors.append(tfb.Permute(permutation=list(reversed(range(9)))))
flow_bijector = tfb.Chain(list(reversed(bijectors)))
model = tfd.TransformedDistribution(
    # distribution=tfd.MultivariateNormalDiag(loc=tf.zeros([args.batch_size, 784])),
    distribution=tfd.Dirichlet(10*tf.ones([args.batch_size, 81, 9])),
    bijector=tfb.MaskedAutoregressiveFlow(
            shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
                hidden_layers=[512, 512])))

# build model
data_ph = tf.placeholder(tf.float32, shape=[args.batch_size, 81, 9])
log_likelihood = model.log_prob(tf.nn.softmax(data_ph))
sample = tf.reshape(model.sample(), [args.batch_size, 9, 9, 9])
global_step = tf.train.get_or_create_global_step()
learning_rate = tf.train.cosine_decay(args.learning_rate, global_step,
                                      args.epochs * batches_per_epoch)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
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

        bits_per_dim = -mean_nll / (np.log(2.) * reduce(operator.mul, data_shape[1:]))
        print("bits per dim: {:7.5f}\tnll: {:7.5f}".format(bits_per_dim, mean_nll))

        #generated_img = session.run(sample_img)
        #for i in range(generated_img.shape[0]):
        #    scipy.misc.toimage(generated_img[i]).save('epoch{}_{}.jpg'.format(epoch, i))
        sample_val = session.run(sample)
        np.save('attempted_sudoku_epoch{}'.format(epoch), sample_val)
