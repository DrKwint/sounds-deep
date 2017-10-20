import numpy as np
import tensorflow as tf
from keras.layers import Flatten

def shape(tensor):
    """
    Return the shape of a tensor as a tuple of int.

    http://stackoverflow.com/a/41771268
    """
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0, len(s))])


def flatten(tensor):
    """
    Flattens a tensor along all non-batch dimensions.

    This is correctly a NOP if the input is already flat.

    Prettytensor has a built-in version of this so two equivalent
    alternatives starting from a vanilla tensor would be

        pt.wrap(tensor).flatten()

    or using this function,

        flatten(tensor)
    """
    if len(shape(tensor)) == 2:
        return tensor
    else:
        return Flatten()(tensor)

def split(x, split_dim, split_sizes):
    n = len(list(x.get_shape()))
    dim_size = np.sum(split_sizes)
    assert int(x.get_shape()[split_dim]) == dim_size
    ids = np.cumsum([0] + split_sizes)
    ids[-1] = -1
    begin_ids = ids[:-1]

    ret = []
    for i in range(len(split_sizes)):
        cur_begin = np.zeros([n], dtype=np.int32)
        cur_begin[split_dim] = begin_ids[i]
        cur_end = np.zeros([n], dtype=np.int32) - 1
        cur_end[split_dim] = split_sizes[i]
        ret += [tf.slice(x, cur_begin, cur_end)]
    return ret

def resize_nearest_neighbor(x, scale):
    input_shape = map(int, x.get_shape().as_list())
    size = [int(input_shape[2] * scale), int(input_shape[3] * scale)]
    x = tf.transpose(x, (0, 2, 3, 1))  # go to NHWC data layout
    x = tf.image.resize_nearest_neighbor(x, size)
    x = tf.transpose(x, (0, 3, 1, 2))  # back to NCHW
    return x
