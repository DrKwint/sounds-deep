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
        session (tf.Session): Session with tf.Graph containing the operations
            passed in `verbose_ops_dict` and `silent_ops`
        steps_per_epoch (int): number of times to run operations
        verbose_ops_dict (dict): strings to tf operations whose values will be
            returned
        feed_dict_fn (callable): called to retrieve the feed_dict
            (dict of tf.placeholder to np.array)
        verbose (bool): whether to use tqdm progressbar on stdout
    Return:
        dict of str to np.array parallel to the verbose_ops_dict
    """
    verbose_vals = {k: [] for k, v in verbose_ops_dict.items()}
    if verbose:
        iterable = tqdm(list(range(steps_per_epoch)))
    else:
        iterable = list(range(steps_per_epoch))

    for step in iterable:
        out = session.run([silent_ops, verbose_ops_dict],
                          feed_dict=feed_dict_fn())[1]
        verbose_vals = {
            k: v + [np.array(out[k])]
            for k, v in verbose_vals.items()
        }

    return {
        k: np.concatenate(v) if v[0].shape != () else np.array(v)
        for k, v in verbose_vals.items()
    }


def train(session,
          epochs,
          train_dict,
          validate_dict,
          train_ops_list,
          verbose_ops_dict,
          exit_fn,
          verbose=True):
    """
    Args:
        session
        epochs
        train_dict
            setup_fn: callable taking session and epoch returning dict to add to verbose ops output
            steps_per_epoch (int)
            feed_dict_fn: callable returning feed_dict
        validate_dict
            setup_fn: callable taking session and epoch returning dict to add to verbose ops output
            steps_per_epoch (int)
            feed_dict_fn: callable returning feed_dict
        train_ops_list: list of operations to silently run in training phase
        verbose_ops_dict: dict of str to tensor to collect results in train and
            validate phases
        exit_fn (Session, int, dict): callable taking session, epoch, and the
            validation verbose ops dict and saving if desired
        verbose (bool): whether to use tqdm progress bar on stdout

    """
    for epoch in range(1, epochs):
        print('TRAIN')
        train_setup_dict = train_dict['setup_fn'](session, epoch)
        train_run_dict = run_epoch_ops(
            session,
            train_dict['steps_per_epoch'],
            verbose_ops_dict=verbose_ops_dict,
            silent_ops=train_ops_list,
            feed_dict_fn=train_dict['feed_dict_fn'],
            verbose=verbose)
        
        train_run_dict = {k: np.mean(v) for k,v in train_run_dict.items()}
        train_val_dict = dict(**train_setup_dict, **train_run_dict)
        print(train_val_dict)

        print('VALIDATE')
        validate_setup_dict = validate_dict['setup_fn'](session, epoch)
        validate_run_dict = run_epoch_ops(
            session,
            validate_dict['steps_per_epoch'],
            verbose_ops_dict=verbose_ops_dict,
            feed_dict_fn=validate_dict['feed_dict_fn'],
            verbose=verbose)

        validate_run_dict = {k: np.mean(v) for k,v in validate_run_dict.items()}
        validate_val_dict = dict(**validate_setup_dict, **validate_run_dict)
        print(validate_val_dict)
        
        # END OF EPOCH
        if exit_fn(session, epoch, validate_val_dict):
            return


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


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


def flatten_sum(logps):
    if len(logps.get_shape()) == 2:
        return tf.reduce_sum(logps, [1])
    elif len(logps.get_shape()) == 4:
        return tf.reduce_sum(logps, [1, 2, 3])
    else:
        raise Exception()


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
