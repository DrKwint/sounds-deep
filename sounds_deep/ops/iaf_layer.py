""" Credit to https://github.com/openai/iaf which was adapted for this """
import sonnet as snt
import tensorflow as tf

from util.shaping import split, resize_nearest_neighbor
from util.distibutions import DiagonalGaussian

class IAFLayer(snt.AbstractModule):
    def __init__(self, h_size, z_size, mode, downsample):
        self._h_size = h_size
        self._z_size = z_size
        self._mode = mode
        self._downsample = downsample
        self._stride = [2, 2] if self._downsample else [1, 1]

    def _build(self):
        pass

class IAFLayerUp(snt.AbstractModule):
    def __init__(self, h_size, z_size, mode, downsample):
        self._h_size = h_size
        self._z_size = z_size
        self._stride = [2,2] if downsample else [1, 1]

    def _build(self, input_t):
        """
        Args:
            input_t

        Returns:
            A 4-tuple of ...            
        """
        x = tf.nn.elu(input_t)
        x = snt.Conv2D(
            output_channels=2 * self._z_size + 2 * self._h_size,
            kernel_shape=(3, 3),
            stride=self._stride)(x)
        qz_mean, qz_logsd, up_context, h = split(
            x, 1, [self._z_size, self._z_size, self._h_size, self._h_size])

        h = tf.nn.elu(h)
        h = snt.Conv2D(
            output_channels=self._h_size, kernel_shape=(3, 3), stride=1)(h)

        if self._downsample:
            input_t = resize_nearest_neighbor(input_t, 0.5)
        return input_t + 0.1 * h, qz_mean, qz_logsd, up_context

class IAFLayerDown(snt.AbstractModule):
    """
    Inverse Autoregressive flow layer using in the inference process
    """
    def __init__(self, h_size, z_size, mode, downsample, batch_size, k, kl_min):
        """
        Args:
            - kl_min (float): free bits/nats
        """
        self._h_size = h_size
        self._z_size = z_size
        self._mode = mode
        self._downsample = downsample
        self._stride = [2, 2] if self._downsample else [1, 1]
        self._batch_size = batch_size
        self._k = k
        self._kl_min = kl_min

    def _build(self, input_t, qz_mean, qz_logsd, up_context):
        x = tf.nn.elu(input_t)
        x = snt.Conv2D(4 * self._z_size, + self._h_size * 2, [3,3], name="down_conv1")
        pz_mean, pz_logsd, rz_mean, rz_logsd, down_context, h_det = split(x, 1, [self._z_size] * 4 + [self._h_size] * 2)

        prior = DiagonalGaussian(pz_mean, pz_logsd)
        posterior = DiagonalGaussian(rz_mean + qz_mean, rz_logsd + qz_logsd)
        context = down_context + up_context

        if self.mode in ["init", "sample"]:
            z = prior()
        else
            z = posterior()

        # calculate kl_cost and kl_obj
        if self.mode == "sample":
            kl_cost = kl_obj = tf.zeros([self._batch_size, self._k])
        else:
            logqs = posterior.log_prob(z)
            ar_stack = ARConvNet2d("ar_stack", context, [self._h_size]*2, [self._z_size]*2)
            x, y = ar_stack(z)
            arw_mean, arw_logsd = 0.1 * x, 0.1 * y
            z = (z - arw_mean) / tf.exp(arw_logsd)
            logqs += arw_logsd
            logps = prior.log_prob(z)
            kl_cost = logqs - logps
            if free_bits > 0:
                kl_ave = tf.reduce_mean(tf.reduce_sum(kl_cost, [2, 3]), [0], keep_dims=True)
                kl_ave = tf.maximum(kl_ave, self._kl_min)
                kl_ave = tf.tile(kl_ave, [self._batch_size * self._k, 1])
                kl_obj = tf.reduce_sum(kl_ave, [1])
            else:
                kl_obj = tf.reduce_sum(kl_cost, [1, 2, 3])
            kl_cost = tf.reduce_sum(kl_cost, [1, 2, 3])

        # calculate output_t
        h = tf.concat(1, [z, h_det])
        h = tf.nn.elu(h)
        if self._downsample:
            input_t = resize_nearest_neighbor(input_t, 2)
            deconv = snt.Conv2DTranspose(h_size, name="down_deconv2")
            h = deconv(h)
        else:
            conv = snt.Conv2D(h_size, name="down_conv2")
            h = conv(h)
        output_t = input_t + (h * 0.1)

        return output_t, kl_obj, kl_cost

class ARConv2d(snt.AbstractModule):
    def __init__(self, name, num_filters, filter_size=(3, 3), stride=(1, 1), pad="SAME", zero_diagonal=True):
        super(ARConv2d, self).__init__(name=name)
        self._num_filters = num_filters
        self._filter_size = filter_size
        self._stride = stride
        self._pad = pad
        self._zero_diagonal = zero_diagonal

    def _build(self, input_t):
        """ TODO: implement data based initialization of parameters as the paper claims it helps """
        h, w = self._filter_size
        n_in = int(input_t.get_shape()[1])
        n_out = self._num_filters

        mask = tf.constant(self._get_conv_ar_mask(h, w, n_in, n_out, self._zero_diagonal))
        masked_conv = conv2d(self._num_filters, self._filter_size, stride=self._stride, padding=self._pad, initializers={'w': tf.random_normal_initializer(0, 0.05)}, mask=mask)
        return masked_conv(input_t)

    def _get_linear_ar_mask(self, n_in, n_out, zero_diagonal=False):
        assert n_in % n_out == 0 or n_out % n_in == 0, "%d - %d" % (n_in, n_out)

        mask = np.ones([n_in, n_out], dtype=np.float32)
        if n_out >= n_in:
            k = n_out / n_in
            for i in range(n_in):
                mask[i + 1:, i * k:(i + 1) * k] = 0
                if zerodiagonal:
                    mask[i:i + 1, i * k:(i + 1) * k] = 0
        else:
            k = n_in / n_out
            for i in range(n_out):
                mask[(i + 1) * k:, i:i + 1] = 0
                if zerodiagonal:
                    mask[i * k:(i + 1) * k:, i:i + 1] = 0
        return mask

    def _get_conv_ar_mask(self, h, w, n_in, n_out, zero_diagonal=False):
        l = (h - 1) / 2
        m = (w - 1) / 2
        mask = np.ones([h, w, n_in, n_out], dtype=tf.float32)
        mask[:l, :, :, :] = 0
        mask[:, :m, :, :] = 0
        mask[l, m, :, :] = self._get_linear_ar_mask(n_in, n_out, zero_diagonal)
        return mask

class ARConvNet2d(snt.AbstractModule):
    def __init__(self, name, context, hidden_sizes, out_sizes):
        """
        Args:
            - hidden_sizes (Iterable of int)
            - out_sizes (Iterable of int)
        """
        super(ARConvNet2d, self).__init__(name=name)
        self._context = context
        self._hsizes = tuple(hidden_sizes)
        self._osizes = tuple(out_sizes)
        self._hlayers = tuple()
        self._olayers = tuple()

        self._instantiate_layers()

    def _instantiate_layers(self):
        with self._enter_variable_scope():
            self._hlayers = tuple(ARConv2d("layer_%d" % i, self._hsizes[i], zero_diagonal=False) for i in xrange(len(self._hsizes)))
            self._olayers = tuple(ARConv2d("layer_%d" % i, self._osizes[i], zero_diagonal=True) for i in xrange(len(self._osizes)))

    def _build(self, x):
        for i, ar_l in enumerate(self._hlayers):
            x = ar_l(x)
            if i == 0:
                x += context
            x = tf.nn.elu(x)
        return map(lambda f: f(x), [self._olayers])

