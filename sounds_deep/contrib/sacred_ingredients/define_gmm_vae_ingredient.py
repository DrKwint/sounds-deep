import numpy as np
import sacred
import sonnet as snt
import tensorflow as tf

import sounds_deep.contrib.data.data as data
import sounds_deep.contrib.models.vae
import sounds_deep.contrib.parameterized_distributions.gmm
import sounds_deep.contrib.util

define_gmm_vae_ingredient = sacred.Ingredient('model')


@define_gmm_vae_ingredient.config
def cfg():
    latent_dimension = 50


@define_gmm_vae_ingredient.capture
def write_verbose_ops(epoch, result_dict, _run):
    result_dict['elbo'] = float(np.mean(result_dict['elbo']))
    result_dict['iw_elbo'] = float(np.mean(result_dict['iw_elbo']))
    _run.info[epoch] = result_dict


@define_gmm_vae_ingredient.capture
def define_model(data_shape, latent_dimension):
    # define model components
    encoder_module = snt.nets.MLP([200, 200])
    decoder_module = snt.nets.MLP([200, 200, 784])

    def prior_fn(latent_dimension):
        cov_init = util.positive_definate_initializer(
            [10] + [latent_dimension] * 2)
        eigvals = tf.self_adjoint_eig(
            tf.divide(
                cov_init + tf.matrix_transpose(cov_init),
                2.,
                name='symmetrised'))[0]
        cov_init = tf.Print(cov_init, [cov_init])

        return parameterized_distributions.gmm.GMM(
            10, latent_dimension, cov_init=cov_init, trainable=True).model

    model = vae.VAE(
        latent_dimension, encoder_module, decoder_module, prior_fn=prior_fn)

    # assemble graph
    input_ph = tf.placeholder(tf.float32, shape=data_shape)
    model(input_ph, n_samples=50, analytic_kl=False)

    # setup optimizer
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(-model.elbo)

    verbose_ops_dict = dict()
    verbose_ops_dict['elbo'] = model.elbo
    verbose_ops_dict['iw_elbo'] = model.importance_weighted_elbo

    return model, input_ph, train_op, verbose_ops_dict
