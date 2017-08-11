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
