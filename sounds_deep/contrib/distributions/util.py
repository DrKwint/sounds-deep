import tensorflow as tf

def logdet(A, name='logdet'):
    """
    Numerically stable implementation of log(det(A)) for symmetric positive definite matrices
    Source: https://github.com/tensorflow/tensorflow/issues/367#issuecomment-176857495
    Args:
        A: positive definite matrix of shape ..., D, D
        name: tf name scope
    Returns:
        log(det(A))
    """
    with tf.name_scope(name):
        # return tf.log(tf.matrix_determinant(A))
        return tf.multiply(
            2., tf.reduce_sum(tf.log(tf.matrix_diag_part(tf.cholesky(A))), axis=-1), name='logdet')
