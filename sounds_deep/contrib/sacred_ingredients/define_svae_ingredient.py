import numpy as np
import sacred
import sonnet as snt
import tensorflow as tf

import data.data as data
import svae

define_svae_ingredient = sacred.Ingredient('model')


@define_svae_ingredient.config
def cfg():
    latent_dimension = 50

@define_svae_ingredient.capture
def write_verbose_ops(epoch, result_dict, _run):
    result_dict['elbo'] = float(np.mean(result_dict['elbo']))
    _run.info[epoch] = result_dict

@define_svae_ingredient.capture
def define_model(data_shape, latent_dimension):
    batch_size = data_shape[0]
    encoder_conv = snt.nets.ConvNet2D(
        [96, 96, 96, 96, 192, 192, 192, 192], [3], [2, 1, 2, 1, 2, 1, 2, 1], [snt.SAME],
        activation=tf.nn.elu,
        regularizers={'w': tf.contrib.layers.l2_regularizer(0.001)})
    encoder_mlp = snt.nets.MLP(
        [latent_dimension * 2], regularizers={'w': tf.contrib.layers.l2_regularizer(0.001)})
    encoder_module = snt.Sequential(
        [encoder_conv, lambda x: tf.reshape(x, [batch_size, -1]), encoder_mlp])
    decoder_mlp = encoder_mlp.transpose()
    decoder_conv = encoder_conv.transpose()
    decoder_module = snt.Sequential(
        [decoder_mlp, lambda x: tf.reshape(x, [-1, 2, 2, 192]), decoder_conv])
    model = svae.GMM_SVAE(latent_dimension, 10, encoder_module, decoder_module)

    # build model
    input_ph = tf.placeholder(tf.float32, shape=data_shape)
    output_distribution, latent_posterior, latent_k_samples, latent_samples, log_z_given_y_phi = model(
        input_ph, nb_samples=20)

    elbo, elbo_details = model.compute_elbo(input_ph, output_distribution, latent_posterior,
                                            latent_k_samples, log_z_given_y_phi)
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(-elbo)

    verbose_ops_dict = dict()
    verbose_ops_dict['elbo'] = elbo

    return model, input_ph, train_op, verbose_ops_dict
