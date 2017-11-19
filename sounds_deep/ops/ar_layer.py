import numpy as np
import tensoflow as tf
import sonnet as snt

def get_conv_ar_mask(h, w, n_in, n_out, zerodiagonal=False):
    """
    Returns:
        Numpy array of shape [h, w, n_in, n_out] with 0 and 1 entries
    """
    l = (h - 1) / 2
    m = (w - 1) / 2
    mask = np.ones([h, w, n_in, n_out], dtype=np.float32)
    mask[:l, :, :, :] = 0
    mask[l, :m, :, :] = 0
    mask[l, m, :, :] = get_linear_ar_mask(n_in, n_out, zerodiagonal)
    return mask

class ARConv2D(snt.AbstractModule):
    def __init__(self, output_channels, kernel_shape=(3,3), stride=(1,1), padding=snt.SAME, zero_diagonal=True, name="ar_conv_2d"):
        super(ARConv2D, self).__init__(name=name)
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._padding = padding
        self._zero_diagonal = zero_diagonal

    def _build(self, x):
        h, w = self._kernel_shape
        n_in = int(x.get_shape()[1])
        n_out = self._output_channels
        mask = get_conv_ar_mask(h, w, n_in, n_out, self._zero_diagonal)
        conv = snt.Conv2D(self._output_channels, self._kernel_shape, self._stride, padding=self._padding, mask=mask)
        return conv(x)

class ARConvNet2D(snt.AbstractModule):
    def __init__(self, h_output_channels, out_output_channels, name="ar_conv_net_2d")
        super(ARConvNet2D, self).__init__(name=name)
        self._h_output_channels = h_output_channels
        self._out_output_channels = out_output_channels

    def _build(self, x, context):
        for i, size in enumerate(self._h_output_channels):
            x = ARConv2D(size, zero_diagonal=False)(x)
            if i == 0:
                x += context
            x = self.activation(x)
        return [ARConv2D(size, zero_diagonal=True)(x) for size in self._out_output_channels]

