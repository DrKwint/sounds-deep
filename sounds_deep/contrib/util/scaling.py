import tensorflow as tf

# 2X nearest-neighbour upsampling, also inspired by Jascha Sohl-Dickstein's code
def upsample2d_nearest_neighbour(x):
    shape = x.get_shape()
    n_batch = int(shape[0])
    height = int(shape[1])
    width = int(shape[2])
    n_channels = int(shape[3])
    x = tf.reshape(x, (n_batch, height, 1, width, 1, n_channels))
    x = tf.concat(2, [x, x])
    x = tf.concat(4, [x, x])
    x = tf.reshape(x, (n_batch, height*2, width*2, n_channels))
    return x


def upsample(x, factor=2):
    shape = x.get_shape()
    height = int(shape[1])
    width = int(shape[2])
    x = tf.image.resize_nearest_neighbor(x, [height * factor, width * factor])
    return x


def squeeze2d(x, factor=2):
    assert factor >= 1
    if factor == 1:
        return x
    shape = x.get_shape()
    height = int(shape[1])
    width = int(shape[2])
    n_channels = int(shape[3])
    assert height % factor == 0 and width % factor == 0
    x = tf.reshape(x, [-1, height//factor, factor,
                       width//factor, factor, n_channels])
    x = tf.transpose(x, [0, 1, 3, 5, 2, 4])
    x = tf.reshape(x, [-1, height//factor, width //
                       factor, n_channels*factor*factor])
    return x


def unsqueeze2d(x, factor=2):
    assert factor >= 1
    if factor == 1:
        return x
    shape = x.get_shape()
    height = int(shape[1])
    width = int(shape[2])
    n_channels = int(shape[3])
    assert n_channels >= 4 and n_channels % 4 == 0
    x = tf.reshape(
        x, (-1, height, width, int(n_channels/factor**2), factor, factor))
    x = tf.transpose(x, [0, 1, 4, 2, 5, 3])
    x = tf.reshape(x, (-1, int(height*factor),
                       int(width*factor), int(n_channels/factor**2)))
    return x
