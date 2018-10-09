import tensorflow as tf

from sounds_deep.contrib.util.util import int_shape

LOGSCALE_FACTOR = 0.01


def actnorm(name,
            x,
            scale=1.,
            logdet=None,
            logscale_factor=LOGSCALE_FACTOR,
            batch_variance=False,
            reverse=False,
            init=False,
            trainable=True):
    if not reverse:
        x = actnorm_center(name + "_center", x, reverse)
        x = actnorm_scale(name + "_scale", x, scale, logdet, logscale_factor,
                          batch_variance, reverse, init)
        if logdet != None:
            x, logdet = x
    else:
        x = actnorm_scale(name + "_scale", x, scale, logdet, logscale_factor,
                          batch_variance, reverse, init)
        if logdet != None:
            x, logdet = x
        x = actnorm_center(name + "_center", x, reverse)
    if logdet != None:
        return x, logdet
    return x


def actnorm_center(name, x, reverse=False):
    shape = x.get_shape()
    with tf.variable_scope(name):
        assert len(shape) == 2 or len(shape) == 4
        if len(shape) == 2:
            x_mean = tf.reduce_mean(x, [0], keepdims=True)
            b = get_variable_ddi(
                "b", (1, int_shape(x)[1]), initial_value=-x_mean)
        elif len(shape) == 4:
            x_mean = tf.reduce_mean(x, [0, 1, 2], keepdims=True)
            b = get_variable_ddi(
                "b", (1, 1, 1, int_shape(x)[3]), initial_value=-x_mean)

        if not reverse:
            x += b
        else:
            x -= b

        return x


def actnorm_scale(name,
                  x,
                  scale=1.,
                  logdet=None,
                  logscale_factor=3.,
                  batch_variance=True,
                  reverse=False,
                  init=False,
                  trainable=True):
    shape = x.get_shape()
    assert len(shape) == 2 or len(shape) == 4
    if len(shape) == 2:
        x_var = tf.reduce_mean(x**2, [0], keepdims=True)
        logdet_factor = 1
        _shape = (1, int_shape(x)[1])

    elif len(shape) == 4:
        x_var = tf.reduce_mean(x**2, [0, 1, 2], keepdims=True)
        logdet_factor = int(shape[1]) * int(shape[2])
        _shape = (1, 1, 1, int_shape(x)[3])

    if batch_variance:
        x_var = tf.reduce_mean(x**2, keepdims=True)

    logs = get_variable_ddi(
        "logs",
        _shape,
        initial_value=tf.log(scale / (tf.sqrt(x_var) + 1e-6)) /
        logscale_factor) * logscale_factor
    if not reverse:
        # logs = tf.Print(logs, [tf.exp(logs)], "actnorm scale: ")
        x = x * tf.exp(logs)
    else:
        x = x * tf.exp(-logs)

    if logdet != None:
        dlogdet = tf.reduce_sum(logs) * logdet_factor
        if reverse:
            dlogdet *= -1
        # dlogdet = tf.Print(dlogdet, [dlogdet], "actnorm dlogdet: ")
        return x, logdet + dlogdet

    return x


def get_variable_ddi(name,
                     shape,
                     initial_value,
                     dtype=tf.float32,
                     init=False,
                     trainable=True):
    w = tf.get_variable(name, shape, dtype, None, trainable=trainable)
    if init:
        w = w.assign(initial_value)
        with tf.control_dependencies([w]):
            return w
    return w
