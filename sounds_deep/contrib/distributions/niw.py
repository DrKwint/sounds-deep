from __future__ import absolute_import, division, print_function

import sonnet as snt
import tensorflow as tf


def standard_to_natural(beta, m, C, v):
    with tf.name_scope('niw_to_nat'):
        K, D = m.get_shape()
        assert beta.get_shape() == (K, )
        D = int(D)

        b = tf.expand_dims(beta, -1) * m
        A = C + _outer(b, m)
        v_hat = v + D + 2

        return A, b, beta, v_hat


def natural_to_standard(A, b, beta, v_hat):
    with tf.name_scope('niw_to_stndrd'):
        m = tf.divide(b, tf.expand_dims(beta, -1))

        K, D = m.get_shape()
        assert beta.get_shape() == (K, )
        D = int(D)

        C = A - _outer(b, m)
        v = v_hat - D - 2
        return beta, m, C, v


def _outer(a, b):
    a_ = tf.expand_dims(a, axis=-1)
    b_ = tf.expand_dims(b, axis=-2)
    return tf.multiply(a_, b_, name='outer')
