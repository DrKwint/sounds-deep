import numpy as np
import sacred
import sonnet as snt
import tensorflow as tf

import data.data as data
import vae

define_vae_ingredient = sacred.Ingredient('model')


@define_vae_ingredient.config
def cfg():
    latent_dimension = 50

@define_vae_ingredient.capture
def write_verbose_ops(epoch, result_dict, _run):
    result_dict['elbo'] = float(np.mean(result_dict['elbo']))
    result_dict['iw_elbo'] = float(np.mean(result_dict['iw_elbo']))
    _run.info[epoch] = result_dict

@define_vae_ingredient.capture
def define_model(data_shape, latent_dimension):
    # define model components
    encoder_module = snt.nets.MLP([200, 200])
    decoder_module = snt.nets.MLP([200, 200, 784])
    model = vae.VAE(latent_dimension, encoder_module, decoder_module)

    # assemble graph
    input_ph = tf.placeholder(tf.float32, shape=data_shape)
    model(input_ph, n_samples=50)

    # setup optimizer
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(-model.elbo)

    verbose_ops_dict = dict()
    verbose_ops_dict['elbo'] = model.elbo
    verbose_ops_dict['iw_elbo'] = model.importance_weighted_elbo

    return model, input_ph, train_op, verbose_ops_dict
