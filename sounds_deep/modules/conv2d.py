""" Conv2D operation, can be masked for Pixel networks """
import numpy as np
import sonnet as snt
import tensorflow as tf


class Conv2D(snt.AbstractModule):
    """ Conv2D operation, can be masked for Pixel networks """
    def __init__(self,
                 filter_num,
                 filter_size,
                 mask_type=None,
                 stride=1,
                 activation_fn=None,
                 use_bias=True,
                 name="conv_2d"):
        """
        Uses square, glorot uniform initialized filters by assumption.

        Args:
            - name (string)
            - filter_num (int): equivalent to number of output channels
            - filter_size (int): side length
            - mask_type ((c, int)): c in {'a', 'b'}
            - stride (int): pixels between filter applications
            - activation_fn (Tensor -> Tensor)
            - use_bias (bool)
        """
        super(Conv2D, self).__init__(name=name)
        self._name = name
        self._filter_num = filter_num
        self._filter_size = filter_size
        self._mask_type = mask_type
        self._stride = stride
        self._activation_fn = activation_fn
        self._use_bias = use_bias

    def _build(self, input):
        """
        Args:

            - input (Tensor): of shape (batch size, num_channels, height, width)
        """
        _, input_dim, _, _ = input.get_shape()
        if self._mask_type is not None:
            mask_type, mask_n_channels = self._mask_type
            mask = np.ones(
                (self._filter_size, self._filter_size, input_dim,
                 self._filter_num),
                dtype='float32')
            center = self._filter_size // 2

            # Mask out future locations
            # filter shape is (height, width, input channels, output channels)
            mask[center + 1:, :, :, :] = 0.
            mask[center, center + 1:, :, :] = 0.

            # Mask out future channels
            for i in xrange(mask_n_channels):
                for j in xrange(mask_n_channels):
                    if (mask_type == 'a' and i >= j) or (mask_type == 'b' and
                                                         i > j):
                        mask[center, center, i::mask_n_channels, j::
                             mask_n_channels] = 0.

            filter_shape = (self._filter_size, self._filter_size, input_dim,
                            self._filter_num)
            initializer = tf.contrib.layers.xavier_initializer_conv2d()
            filters = tf.get_variable(
                self._name + '.filters',
                shape=filter_shape,
                initializer=initializer)
            if mask_type is not None:
                filters = filters * mask

            result = tf.nn.conv2d(
                input=input,
                filter=filters,
                strides=[1, 1, self._stride, self._stride],
                padding='SAME',
                data_format='NCHW')

            if self._use_bias:
                biases = tf.get_variable(
                    name=self._name + '.biases',
                    shape=filter_shape,
                    initializer=tf.zeros_initializer())
                result = tf.nn.bias_add(result, biases)

            if self._activation_fn is not None:
                result = self._activation_fn(result)

            return result
