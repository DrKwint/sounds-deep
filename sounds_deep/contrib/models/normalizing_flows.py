import numpy as np
import sonnet as snt
import tensorflow as tf

from sounds_deep.contrib.util.util import int_shape
from sounds_deep.contrib.util.actnorm import actnorm
from sounds_deep.contrib.util.scaling import squeeze2d, unsqueeze2d

tfd = tf.contrib.distributions

def conv2d_zeros(name,
                 x,
                 width,
                 filter_size=[3, 3],
                 stride=[1, 1],
                 pad="SAME",
                 logscale_factor=3):
    with tf.variable_scope(name):
        n_in = int(x.get_shape()[3])
        stride_shape = [1] + stride + [1]
        filter_shape = filter_size + [n_in, width]
        w = tf.get_variable(
            "W", filter_shape, tf.float32, initializer=tf.zeros_initializer())
        x = tf.nn.conv2d(x, w, stride_shape, pad, data_format='NHWC')
        x += tf.get_variable(
            "b", [1, 1, 1, width], initializer=tf.zeros_initializer())
        x *= tf.exp(
            tf.get_variable(
                "logs", [1, width], initializer=tf.zeros_initializer()) *
            logscale_factor)
    return x


def linear_zeros(name, x, width, logscale_factor=3):
    with tf.variable_scope(name):
        n_in = int(x.get_shape()[1])
        w = tf.get_variable(
            "W", [n_in, width], tf.float32, initializer=tf.zeros_initializer())
        x = tf.matmul(x, w)
        x += tf.get_variable(
            "b", [1, width], initializer=tf.zeros_initializer())
        x *= tf.exp(
            tf.get_variable(
                "logs", [1, width], initializer=tf.zeros_initializer()) *
            logscale_factor)
        return x


class GaussianizeSplit(snt.AbstractModule):
    def __init__(self, name='gaussian_split'):
        super(GaussianizeSplit, self).__init__(name=name)

    def _get_prior(self, z):
        h = conv2d_zeros("conv", z, 2 * int(z.get_shape()[3]))
        mean = h[:, :, :, 0::2]
        logs = h[:, :, :, 1::2]
        return tfd.MultivariateNormalDiag(loc=mean, scale_diag=tf.exp(logs))

    def _build(self, z, log_prob=0., reverse=False, eps=None):
        if not reverse:
            z1, z2 = tf.split(z, 2, axis=-1)
            prior = self._get_prior(z1)
            log_prob += prior.log_prob(z2)
            z1 = squeeze2d(z1)
            eps = (z2 - prior.loc) / prior.scale
            return z1, log_prob, eps
        else:
            z1 = unsqueeze2d(z)
            prior = self._get_prior(z1)
            if eps is not None:
                z2 = (eps * prior.scale) + prior.loc
            else:
                z2 = prior.sample()
            z = tf.concat([z1, z2], -1)
            return z


class Invertible1x1Conv(snt.AbstractModule):
    def __init__(self, name="invertible_1x1_conv"):
        super(Invertible1x1Conv, self).__init__(name=name)

    def _build(self, z, logdet, reverse=False):
        """
        Args:
        z: NHWC
        """
        shape = int_shape(z)
        w_shape = [shape[3], shape[3]]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype('float32')
        w = tf.get_variable("W", dtype=tf.float32, initializer=w_init)
        dlogdet = tf.cast(
            tf.log(abs(tf.matrix_determinant(tf.cast(w, 'float64')))),
            'float32') * shape[1] * shape[2]

        if not reverse:
            _w = tf.reshape(w, [1, 1] + w_shape)
            z = tf.nn.conv2d(z, _w, [1, 1, 1, 1], 'SAME', data_format='NHWC')
            logdet += dlogdet
        else:
            _w = tf.matrix_inverse(w)
            _w = tf.reshape(_w, [1, 1] + w_shape)
            z = tf.nn.conv2d(z, _w, [1, 1, 1, 1], 'SAME', data_format='NHWC')
            logdet -= dlogdet

        return z, logdet


def glow_net_fn(width=512, n_out=None):
    layers = []
    layers.append(snt.Conv2D(width, [3, 3]))
    layers.append(tf.nn.relu)
    layers.append(snt.Conv2D(width, [1, 1]))
    layers.append(tf.nn.relu)
    layers.append(lambda x: conv2d_zeros("conv2d_zeros", x, n_out))
    return snt.Sequential(layers)


class FlowCoupling(snt.AbstractModule):
    def __init__(self, net_fn, action="scale_and_shift", name="flow_coupling"):
        super(FlowCoupling, self).__init__(name=name)
        self.net_fn = net_fn
        self.action = action
        assert action in ['shift', 'scale_and_shift']

    def _build(self, z, logdet, reverse=False):
        z_rank = len(z.get_shape())
        z1, z2 = tf.split(z, 2, axis=-1)

        # calculate scale, shift
        if self.action == "shift":
            n_out = int(z1.get_shape()[3])
            shift = self.net_fn(n_out=n_out)(z1)
            scale = tf.ones_like(shift)
        else:
            n_out = 2 * int(z1.get_shape()[3])
            shift, scale = tf.split(self.net_fn(n_out=n_out)(z1), 2, axis=-1)
            scale = tf.nn.sigmoid(scale + 2.)
        local_logdet = tf.reduce_sum(
            tf.log(scale), axis=list(range(1, z_rank)))

        # apply scale, shift
        if not reverse:
            z2 += shift
            z2 *= scale
            logdet += local_logdet
        else:
            z2 /= scale
            z2 -= shift
            logdet -= local_logdet

        z = tf.concat([z1, z2], 3)
        return z, logdet


class RevNet2dStep(snt.AbstractModule):
    def __init__(self, net_fn, flow_coupling_type, name="revnet_2d_step"):
        super(RevNet2dStep, self).__init__(name=name)

        with self._enter_variable_scope():
            self.invertible_1x1_conv = Invertible1x1Conv()
            self.flow_coupling = FlowCoupling(net_fn, flow_coupling_type)

    def _build(self, z, logdet, reverse=False):
        assert int_shape(z)[3] % 2 == 0
        if not reverse:
            z, logdet = actnorm("actnorm", z, logdet=logdet)
            z, logdet = self.invertible_1x1_conv(z, logdet)
            z, logdet = self.flow_coupling(z, logdet)
        else:
            z, logdet = self.flow_coupling(z, logdet, reverse=True)
            z, logdet = self.invertible_1x1_conv(z, logdet, reverse=True)
            z, logdet = actnorm("actnorm", z, logdet=logdet, reverse=True)
        return z, logdet


class GlowFlow(snt.AbstractModule):
    def __init__(self,
                 n_levels,
                 depth_per_level,
                 net_fn,
                 flow_coupling_type,
                 name="glow_flow"):
        super(GlowFlow, self).__init__(name=name)

        with self._enter_variable_scope():
            self.revnets = []
            self.splits = []
            for i in range(n_levels):
                level_flows = []
                for k in range(depth_per_level):
                    level_flows.append(
                        RevNet2dStep(net_fn, flow_coupling_type))
                self.revnets.append(snt.Sequential(level_flows))
                self.splits.append(GaussianizeSplit())

    def _build(self, z, logdet, reverse=False, eps=None):
        if not reverse:
            eps = []
            for i, revnet in enumerate(self.revnets):
                z, logdet = revnet(z, logdet)
                if i < len(self.revnets) - 1:
                    z, logdet, local_eps = self.splits[i](z, logdet)
                    eps.append(local_eps)
            return z, logdet, eps
        else:
            if eps is None: eps = [None] * len(self.revnets)
            for i, revnet in reversed(list(enumerate(self.revnets))):
                if i < len(self.revnets) - 1:
                    z = self.splits[i](z, reverse=True)
                z, _ = revnet(z, 0., reverse=True, eps=eps[i])
            return z


class NormalizingFlows(snt.AbstractModule):
    def __init__(self, flow, name="normalizing_flows"):
        super(NormalizingFlows, self).__init__(name=name)
        self.flow = flow

    def _preprocess(self, x, n_bits_x, n_bins):
        x = tf.cast(x, 'float32')
        if n_bits_x < 8:
            x = tf.floor(x / 2**(8 - n_bits_x))
        x = x / n_bins - .5
        return x

    def _postprocess(self, x, n_bins):
        return tf.cast(
            tf.clip_by_value(
                tf.floor((x + .5) * n_bins) * (256. / n_bins), 0, 255),
            'uint8')

    def _prior(self, y_onehot, top_shape, learn_top=True, ycond=True):
        n_z = top_shape[-1]

        h = tf.zeros([tf.shape(y_onehot)[0]] + top_shape[:2] + [2 * n_z])
        if learn_top:
            h = conv2d_zeros('p', h, 2 * n_z)
        if ycond:
            h += tf.reshape(
                linear_zeros("y_emb", y_onehot, 2 * n_z), [-1, 1, 1, 2 * n_z])

        pz = tfd.Independent(tfd.MultivariateNormalDiag(
            loc=h[:, :, :, :n_z], scale_diag=tf.exp(h[:, :, :, n_z:])))
        return pz

    def _build(self, x, y_onehot, n_bits_x=8, weight_y=1.0):
        n_bins = 2.**n_bits_x
        n_y = int_shape(y_onehot)[-1]

        # Discrete -> Continuous
        objective = tf.zeros_like(x, dtype='float32')[:, 0, 0, 0]
        z = self._preprocess(x, n_bits_x, n_bins)
        z = z + tf.random_uniform(tf.shape(z), 0, 1. / n_bins)
        objective += -np.log(n_bins) * np.prod(int_shape(z)[1:])

        # Encode
        z = squeeze2d(z, 2)  # > 16x16x12
        z, objective, _ = self.flow(z, objective)

        # Prior
        self.top_shape = int_shape(z)[1:]
        pz = self._prior(y_onehot, self.top_shape)
        objective += pz.log_prob(z)

        # Generative loss
        nobj = -objective
        bits_x = nobj / (np.log(2.) * int(x.get_shape()[1]) * int(
            x.get_shape()[2]) * int(x.get_shape()[3]))  # bits per subpixel

        # Predictive loss
        if True:  # ycond:
            # Classification loss
            h_y = tf.reduce_mean(z, axis=[1, 2])
            y_logits = linear_zeros("classifier", h_y, n_y)
            bits_y = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=y_onehot, logits=y_logits) / np.log(2.)

            # Classification accuracy
            y_predicted = tf.argmax(y_logits, 1, output_type=tf.int32)
            y_true = tf.argmax(y_onehot, 1, output_type=tf.int32)
            classification_error = 1 - \
                tf.cast(tf.equal(y_predicted, y_true), tf.float32)
        else:
            bits_y = tf.zeros_like(bits_x)
            classification_error = tf.ones_like(bits_x)

        loss = bits_x + weight_y * bits_y
        stats_dict = {
            'loss': loss,
            'bits_x': bits_x,
            'bits_y': bits_y,
            'classification_error': classification_error
        }

        return tf.reduce_mean(loss), stats_dict

    def sample(self, y_onehot, eps_std, n_bits_x=8):
        n_bins = 2.**n_bits_x
        y_onehot = tf.cast(y_onehot, 'float32')
        pz = self._prior(y_onehot, self.top_shape)
        z = pz.sample()
        z = self.flow(z, reverse=True)
        z = unsqueeze2d(z, 2)  # 8x8x12 -> 16x16x3
        x = self._postprocess(z, n_bins)
        return x
