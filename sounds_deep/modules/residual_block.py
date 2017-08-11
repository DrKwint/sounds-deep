import functools

import sonnet as snt
import tensorflow as tf

from conv2d import Conv2D


class ResidualBlock(snt.AbstractModule):
    def __init__(self,
                 output_dim,
                 filter_size,
                 mask_type=None,
                 resample=None,
                 name='residual_block'fi):
        """
        Args:
            - name (str)
            - output_dim (int): number of channels in output tensor
            - filter_size (int): side length of a conv filter
            - mask_type ((c, int), optional): where c in {'a', 'b'}
            - resample (str, optional) in {None, 'down', 'up'}
        """
        if resample not in [None, 'down', 'up']:
            raise Exception('Unsupported configuration')
        if mask_type != None and resample != None:
            raise Exception('Unsupported configuration')

        super(ResidualBlock, self).__init__(name=name)
        self._name = name
        self._output_dim = output_dim
        self._filter_size = filter_size
        self._mask_type = mask_type
        self._resample = resample

    def _subpixel_conv2d(self, *args, **kwargs):
        # Creates and builds a `Conv2D` for upscaling
        kwargs['output_dim'] = 4 * kwargs['output_dim']
        output = Conv2D(*args, **kwargs)(kwargs['input'])
        output = tf.transpose(output, [0, 2, 3, 1])
        output = tf.depth_to_space(output, 2)
        output = tf.transpose(output, [0, 3, 1, 2])
        return output

    def _nonlinearity(self, x):
        return tf.nn.elu(x)

    def _pixelcnn_gated_nonlinearity(self, a, b):
        return tf.sigmoid(a) * tf.tanh(b)

    def _build(self, _input):
        """
        Args:

            - input (Tensor): of shape (batch size, num_channels, height, width)
        """
        _, input_dim, _, _ = _input.get_shape()

        if self._resample == 'down':
            conv_shortcut = functools.partial(Conv2D, stride=2)
            conv_1 = functools.partial(Conv2D, output_dim=input_dim)
            conv_2 = functools.partial(
                Conv2D, output_dim=self._output_dim, stride=2)
        elif self._resample == 'up':
            conv_shortcut = self._subpixel_conv2d
            conv_1 = functools.partial(
                self._subpixel_conv2d, output_dim=self._output_dim)
            conv_2 = functools.partial(Conv2D, output_dim=self._output_dim)
        else:
            conv_shortcut = Conv2D
            conv_1 = functools.partial(Conv2D, output_dim=self._output_dim)
            conv_2 = functools.partial(Conv2D, output_dim=self._output_dim)

        if self._output_dim == input_dim and self._resample is None:
            shortcut = _input  # Identity skip-connection
        else:
            shortcut = conv_shortcut(
                self._name + '.shortcut',
                output_dim=self._output_dim,
                filter_size=1,
                mask_type=self._mask_type)

        output = _input
        if self._mask_type is None:
            output = self._nonlinearity(output)
            output = conv_1(
                self._name + '.conv1',
                filter_size=self._filter_size,
                mask_type=self._mask_type)
            output = self._nonlinearity(output)
            output = conv_2(
                self._name + '.conv2',
                filter_size=self._filter_size,
                mask_type=self._mask_type,
                use_bias=False)
            output = Batchnorm(self._name + '.BN', [0, 2, 3], output,
                               bn_is_training, bn_stats_iter)
        else:
            output = self._nonlinearity(output)
            output_a = conv_1(
                self._name + '.conv1A',
                filter_size=self._filter_size,
                mask_type=self._mask_type)
            output_b = conv_1(
                self._name + '.conv1B',
                filter_size=self._filter_size,
                mask_type=self._mask_type)
            output = self._pixelcnn_gated_nonlinearity(output_a, output_b)
            output = conv_2(
                self._name + '.conv2',
                filter_size=self._filter_size,
                mask_type=self._mask_type)

        return shortcut + output
