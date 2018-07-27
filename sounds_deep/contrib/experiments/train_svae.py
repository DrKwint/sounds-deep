import argparse

import numpy as np
import sonnet as snt
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sounds_deep.contrib.data.data as data
import sounds_deep.contrib.models.svae
import sounds_deep.contrib.util
# import visualise_gmm

parser = argparse.ArgumentParser(description='Train an SVAE model.')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--latent_dimension', type=int, default=2)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--bn_step_size', type=float, default=0.15)
args = parser.parse_args()

# load the data
train_data, _, _, _ = data.load_cifar10('./data/')
train_data = train_data[:args.batch_size * 20]
data_shape = (args.batch_size, ) + train_data.shape[1:]
num_pixels = np.prod(train_data.shape[-3:])
print("Num Pixels: " + str(num_pixels))
train_gen = data.data_generator(train_data, args.batch_size)

# build the model
encoder_conv = snt.nets.ConvNet2D(
    [96, 96, 96, 96, 192, 192, 192, 192], [3], [2, 1, 1, 2, 1, 1, 2, 1], [snt.SAME],
    activation=tf.nn.elu,
    regularizers={'w': tf.contrib.layers.l2_regularizer(0.001)})
encoder_mlp = snt.nets.MLP(
    [args.latent_dimension * 2], regularizers={'w': tf.contrib.layers.l2_regularizer(0.001)})
encoder_module = snt.Sequential(
    [encoder_conv, lambda x: tf.reshape(x, [args.batch_size, -1]), encoder_mlp])
decoder_mlp = encoder_mlp.transpose()
decoder_conv = encoder_conv.transpose()
decoder_module = snt.Sequential(
    [decoder_mlp, lambda x: tf.reshape(x, [-1, 4, 4, 192]), decoder_conv])
model = svae.GMM_SVAE(args.latent_dimension, 10, encoder_module, decoder_module)

# build model
data_ph = tf.placeholder(tf.float32, shape=data_shape)
output_distribution, latent_posterior, latent_k_samples, latent_samples, log_z_given_y_phi = model(
    data_ph, nb_samples=20)

elbo, elbo_details = model.compute_elbo(data_ph, output_distribution, latent_posterior,
                                        latent_k_samples, log_z_given_y_phi)
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(-elbo)

# update GMM
update_theta_op = model.m_step_op(
    latent_posterior_samples=latent_samples,
    r_nk=tf.exp(log_z_given_y_phi),
    step_size=args.bn_step_size)

training_step = tf.group(update_theta_op, train_op, name='training_ops')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

verbose_ops_dict = dict()
verbose_ops_dict['elbo'] = -elbo
verbose_ops_dict['reconstruction_loss'] = elbo_details[0]
verbose_ops_dict['weighted_log_numerator'] = elbo_details[1]
verbose_ops_dict['weighted_log_denominator'] = elbo_details[2]
verbose_ops_dict['regularizer_term'] = elbo_details[3]
# verbose_ops_dict['grads'] = {k: tf.global_norm([v]) for k,v in optimizer.compute_gradients(-elbo)}

with tf.Session(config=config) as session:
    session.run(tf.global_variables_initializer())

    fig, ax = plt.subplots(tight_layout=True)
    pi, mu_k, sigma_k = session.run(model.phi_gmm.standard_parameters())
    visualise_gmm.plot_components(mu_k, sigma_k, pi, ax)
    fig.savefig('initial_clusters.png')

    for epoch in range(args.epochs):
        print("EPOCH {}".format(epoch))
        out_dict = util.run_epoch_ops(
            session,
            train_data.shape[0] // args.batch_size,
            silent_ops=[training_step],
            verbose_ops_dict=verbose_ops_dict,
            feed_dict_fn=lambda: {data_ph: next(train_gen)},
            verbose=True)

        mean_elbo = np.mean(out_dict['elbo'])
        recon_error = np.mean(out_dict['reconstruction_loss'])
        weighted_log_numerator = np.mean(out_dict['weighted_log_numerator'])
        weighted_log_denominator = np.mean(out_dict['weighted_log_denominator'])
        regularizer = np.mean(out_dict['regularizer_term'])
        bits_per_dim = mean_elbo / (np.log(2.) * num_pixels)
        print(
            'BITS/DIM: {:.3f}\tELBO: {:.3f}\tRecon: {:.3f}\tRegularizer: {:.3f}\tlog_numerator: {:.3f}\tlog_denominator: {:.3f}'.
            format(bits_per_dim, mean_elbo, recon_error, regularizer, weighted_log_numerator,
                   weighted_log_denominator))
        # print(out_dict['grads'])

        fig, ax = plt.subplots(tight_layout=True)
        pi, mu_k, sigma_k = session.run(model.phi_gmm.standard_parameters())
        print(pi)
        print(mu_k)
        print(sigma_k)
        visualise_gmm.plot_components(mu_k, sigma_k, pi, ax)
        pi, mu_k, sigma_k = session.run(model.theta.standard_parameters())
        print(pi)
        print(mu_k)
        print(sigma_k)
        fig.savefig('epoch_{}_clusters.png'.format(epoch))
