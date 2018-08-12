import numpy as np
import tensorflow as tf
from tqdm import tqdm


def run_epoch_ops(session,
                  steps_per_epoch,
                  verbose_ops_dict={},
                  silent_ops=[],
                  feed_dict_fn=lambda: None,
                  verbose=False):
    """
    Args:
        session (tf.Session): Session containing the operations passed in `verbose_ops_dict` and `silent_ops`
        steps_per_epoch (int): number of times to run operations
        verbose_ops_dict (dict): strings to tf operations whose values will be returned
        feed_dict_fn (callable): called to retrieve the feed_dict
                                   (dict of placeholders to np arrays)
        verbose (bool): whether to use tqdm progressbar on stdout
    Return:
        dict of str to numpy arrays or floats
    """
    verbose_vals = {k: [] for k, v in verbose_ops_dict.items()}
    if verbose:
        iterable = tqdm(list(range(steps_per_epoch)))
    else:
        iterable = list(range(steps_per_epoch))

    for step in iterable:
        out = session.run(
            [silent_ops, verbose_ops_dict], feed_dict=feed_dict_fn())[1]
        verbose_vals = {k: v + [np.array(out[k])] for k, v in verbose_vals.items()}

    return {
        k: np.concatenate(v) if v[0].shape != () else np.array(v)
        for k, v in verbose_vals.items()
    }

def logdet(A, name='logdet'):
    """
    Numerically stable implementation of log(det(A)) for symmetric positive definite matrices

    Source: https://github.com/tensorflow/tensorflow/issues/367#issuecomment-176857495

    Args:
        A: positive definite matrix of shape [..., D, D]
        name: tf name scope
    Returns:
        log(det(A))
    """
    with tf.name_scope(name):
        # return tf.log(tf.matrix_determinant(A))
        return tf.multiply(
            2.,
            tf.reduce_sum(
                tf.log(tf.matrix_diag_part(tf.cholesky(A))), axis=-1),
            name='logdet')

def matrix_is_pos_def_op(A):
    eigvals = tf.self_adjoint_eig(
        tf.divide(A + tf.matrix_transpose(A), 2., name='symmetrised'))[0]
    return tf.assert_positive(
        eigvals, message='Matrix is not positive definite')


def positive_definate_initializer(shape, dtype=tf.float32):
    rows, cols = shape[-2:]
    vals = tf.random_normal(shape, mean=0.0, stddev=0.01, dtype=dtype)
    vals += tf.transpose(vals, perm=[0, 2, 1])
    eye = tf.eye(rows, cols, dtype=dtype)
    return vals + eye


def int_shape(x):
    if str(x.get_shape()[0]) != '?':
        return list(map(int, x.get_shape()))
    return [-1] + list(map(int, x.get_shape()[1:]))


def shuffle_features(name,
                     h,
                     indices=None,
                     return_indices=False,
                     reverse=False):
    with tf.variable_scope(name):

        rng = np.random.RandomState(
            (abs(hash(tf.get_variable_scope().name))) % 10000000)

        if indices == None:
            # Create numpy and tensorflow variables with indices
            n_channels = int(h.get_shape()[-1])
            indices = list(range(n_channels))
            rng.shuffle(indices)
            # Reverse it
            indices_inverse = [0] * n_channels
            for i in range(n_channels):
                indices_inverse[indices[i]] = i

        tf_indices = tf.get_variable(
            "indices",
            dtype=tf.int32,
            initializer=np.asarray(indices, dtype='int32'),
            trainable=False)
        tf_indices_reverse = tf.get_variable(
            "indices_inverse",
            dtype=tf.int32,
            initializer=np.asarray(indices_inverse, dtype='int32'),
            trainable=False)

        _indices = tf_indices
        if reverse:
            _indices = tf_indices_reverse

        if len(h.get_shape()) == 2:
            # Slice
            h = tf.transpose(h)
            h = tf.gather(h, _indices)
            h = tf.transpose(h)
        elif len(h.get_shape()) == 4:
            # Slice
            h = tf.transpose(h, [3, 1, 2, 0])
            h = tf.gather(h, _indices)
            h = tf.transpose(h, [3, 1, 2, 0])
        if return_indices:
            return h, indices
        return h