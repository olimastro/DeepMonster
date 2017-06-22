"""
neural network stuff, intended to be used with Lasagne

All this code, except otherwise mentionned,
was written by openai taken from improvedgan repo on github
"""

import numpy as np
import theano as th
import theano.tensor as T
from theano.tensor.nnet.abstract_conv import (bilinear_upsampling, )
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import lasagne
from lasagne.layers import dnn, ElemwiseSumLayer, NonlinearityLayer
from lasagne.init import Normal, Constant

from deepmonster.adlf.utils import parse_tuple

# T.nnet.relu has some stability issues, this is better
def relu(x):
    return T.maximum(x, 0)

def lrelu(x, a=0.2):
    return T.maximum(x, a*x)

# kyle's relu
def divrelu(x):
    return (x + abs(x)) / 2.

# kyle's tanh
def hard_tanh(x):
    return T.clip(x, -1., 1.)

def centered_softplus(x):
    return T.nnet.softplus(x) - np.cast[th.config.floatX](np.log(2.))

def log_sum_exp(x, axis=1):
    m = T.max(x, axis=axis, keepdims=True)
    return m+T.log(T.sum(T.exp(x-m), axis=axis) + 1e-9)

def adam_updates(params, cost, lr=0.001, mom1=0.9, mom2=0.999):
    updates = []
    grads = T.grad(cost, params)
    t = th.shared(np.cast[th.config.floatX](1.))
    for p, g in zip(params, grads):
        v = th.shared(np.cast[th.config.floatX](p.get_value() * 0.))
        mg = th.shared(np.cast[th.config.floatX](p.get_value() * 0.))
        v_t = mom1*v + (1. - mom1)*g
        mg_t = mom2*mg + (1. - mom2)*T.square(g)
        v_hat = v_t / (1. - mom1 ** t)
        mg_hat = mg_t / (1. - mom2 ** t)
        g_t = v_hat / T.sqrt(mg_hat + 1e-8)
        p_t = p - lr * g_t
        updates.append((v, v_t))
        updates.append((mg, mg_t))
        updates.append((p, p_t))
    updates.append((t, t+1))
    return updates

class WeightNormLayer(lasagne.layers.Layer):
    def __init__(self, incoming, b=lasagne.init.Constant(0.), g=lasagne.init.Constant(1.),
                 W=lasagne.init.Normal(0.05), train_g=False, init_stdv=1., nonlinearity=relu, **kwargs):
        super(WeightNormLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = nonlinearity
        self.init_stdv = init_stdv
        k = self.input_shape[1]
        if b is not None:
            self.b = self.add_param(b, (k,), name="b", regularizable=False)
        if g is not None:
            self.g = self.add_param(g, (k,), name="g", regularizable=False, trainable=train_g)
        if len(self.input_shape)==4:
            self.axes_to_sum = (0,2,3)
            self.dimshuffle_args = ['x',0,'x','x']
        else:
            self.axes_to_sum = 0
            self.dimshuffle_args = ['x',0]

        # scale weights in layer below
        incoming.W_param = incoming.W
        #incoming.W_param.set_value(W.sample(incoming.W_param.get_value().shape))
        if incoming.W_param.ndim==4:
            if isinstance(incoming, Deconv2DLayer):
                W_axes_to_sum = (0,2,3)
                W_dimshuffle_args = ['x',0,'x','x']
            else:
                W_axes_to_sum = (1,2,3)
                W_dimshuffle_args = [0,'x','x','x']
        else:
            W_axes_to_sum = 0
            W_dimshuffle_args = ['x',0]
        if g is not None:
            incoming.W = incoming.W_param * (self.g/T.sqrt(1e-6 + T.sum(T.square(incoming.W_param),axis=W_axes_to_sum))).dimshuffle(*W_dimshuffle_args)
        else:
            incoming.W = incoming.W_param / T.sqrt(1e-6 + T.sum(T.square(incoming.W_param),axis=W_axes_to_sum,keepdims=True))

    def get_output_for(self, input, init=False, **kwargs):
        if init:
            m = T.mean(input, self.axes_to_sum)
            input -= m.dimshuffle(*self.dimshuffle_args)
            inv_stdv = self.init_stdv/T.sqrt(T.mean(T.square(input), self.axes_to_sum))
            input *= inv_stdv.dimshuffle(*self.dimshuffle_args)
            self.init_updates = [(self.b, -m*inv_stdv), (self.g, self.g*inv_stdv)]
        elif hasattr(self,'b'):
            input += self.b.dimshuffle(*self.dimshuffle_args)

        return self.nonlinearity(input)

def weight_norm(layer, **kwargs):
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = lasagne.nonlinearities.identity
    if hasattr(layer, 'b'):
        del layer.params[layer.b]
        layer.b = None
    return WeightNormLayer(layer, nonlinearity=nonlinearity, **kwargs)

class Deconv2DLayer(lasagne.layers.Layer):
    def __init__(self, incoming, target_shape, filter_size, stride=(2, 2), pad='half',
                 W=lasagne.init.Normal(0.05), b=lasagne.init.Constant(0.), nonlinearity=relu, **kwargs):
        super(Deconv2DLayer, self).__init__(incoming, **kwargs)
        self.target_shape = target_shape
        self.nonlinearity = (lasagne.nonlinearities.identity if nonlinearity is None else nonlinearity)
        self.filter_size = lasagne.layers.dnn.as_tuple(filter_size, 2)
        self.stride = lasagne.layers.dnn.as_tuple(stride, 2)
        self.pad = pad

        self.W_shape = (incoming.output_shape[1], target_shape[1], filter_size[0], filter_size[1])
        self.W = self.add_param(W, self.W_shape, name="W")
        if b is not None:
            self.b = self.add_param(b, (target_shape[1],), name="b")
        else:
            self.b = None

    def get_output_for(self, input, **kwargs):
        op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(
            imshp=self.target_shape, kshp=self.W_shape, subsample=self.stride, border_mode=self.pad)
        activation = op(self.W, input, self.target_shape[2:])

        if self.b is not None:
            activation += self.b.dimshuffle('x', 0, 'x', 'x')

        return self.nonlinearity(activation)

    def get_output_shape_for(self, input_shape):
        return self.target_shape

# minibatch discrimination layer
class MinibatchLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_kernels, dim_per_kernel=5, theta=lasagne.init.Normal(0.05),
                 log_weight_scale=lasagne.init.Constant(0.), b=lasagne.init.Constant(-1.), **kwargs):
        super(MinibatchLayer, self).__init__(incoming, **kwargs)
        self.num_kernels = num_kernels
        num_inputs = int(np.prod(self.input_shape[1:]))
        self.theta = self.add_param(theta, (num_inputs, num_kernels, dim_per_kernel), name="theta")
        self.log_weight_scale = self.add_param(log_weight_scale, (num_kernels, dim_per_kernel), name="log_weight_scale")
        self.W = self.theta * (T.exp(self.log_weight_scale)/T.sqrt(T.sum(T.square(self.theta),axis=0))).dimshuffle('x',0,1)
        self.b = self.add_param(b, (num_kernels,), name="b")

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], np.prod(input_shape[1:])+self.num_kernels)

    def get_output_for(self, input, init=False, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        activation = T.tensordot(input, self.W, [[1], [0]])
        abs_dif = (T.sum(abs(activation.dimshuffle(0,1,2,'x') - activation.dimshuffle('x',1,2,0)),axis=2)
                    + 1e6 * T.eye(input.shape[0]).dimshuffle(0,'x',1))

        if init:
            mean_min_abs_dif = 0.5 * T.mean(T.min(abs_dif, axis=2),axis=0)
            abs_dif /= mean_min_abs_dif.dimshuffle('x',0,'x')
            self.init_updates = [(self.log_weight_scale, self.log_weight_scale-T.log(mean_min_abs_dif).dimshuffle(0,'x'))]

        f = T.sum(T.exp(-abs_dif),axis=2)

        if init:
            mf = T.mean(f,axis=0)
            f -= mf.dimshuffle('x',0)
            self.init_updates.append((self.b, -mf))
        else:
            f += self.b.dimshuffle('x',0)

        return T.concatenate([input, f], axis=1)

class BatchNormLayer(lasagne.layers.Layer):
    def __init__(self, incoming, b=lasagne.init.Constant(0.), g=lasagne.init.Constant(1.), nonlinearity=relu, **kwargs):
        super(BatchNormLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = nonlinearity
        k = self.input_shape[1]
        if b is not None:
            self.b = self.add_param(b, (k,), name="b", regularizable=False)
        if g is not None:
            self.g = self.add_param(g, (k,), name="g", regularizable=False)
        self.avg_batch_mean = self.add_param(lasagne.init.Constant(0.), (k,), name="avg_batch_mean", regularizable=False, trainable=False)
        self.avg_batch_var = self.add_param(lasagne.init.Constant(1.), (k,), name="avg_batch_var", regularizable=False, trainable=False)
        if len(self.input_shape)==4:
            self.axes_to_sum = (0,2,3)
            self.dimshuffle_args = ['x',0,'x','x']
        else:
            self.axes_to_sum = 0
            self.dimshuffle_args = ['x',0]

    def get_output_for(self, input, deterministic=False, set_bn_updates=True, **kwargs):
        if deterministic:
            norm_features = (input-self.avg_batch_mean.dimshuffle(*self.dimshuffle_args)) / T.sqrt(1e-6 + self.avg_batch_var).dimshuffle(*self.dimshuffle_args)
        else:
            batch_mean = T.mean(input,axis=self.axes_to_sum).flatten()
            centered_input = input-batch_mean.dimshuffle(*self.dimshuffle_args)
            batch_var = T.mean(T.square(centered_input),axis=self.axes_to_sum).flatten()
            batch_stdv = T.sqrt(1e-6 + batch_var)
            norm_features = centered_input / batch_stdv.dimshuffle(*self.dimshuffle_args)

            # BN updates
            if set_bn_updates:
                new_m = 0.9*self.avg_batch_mean + 0.1*batch_mean
                new_v = 0.9*self.avg_batch_var + T.cast((0.1*input.shape[0])/(input.shape[0]-1),th.config.floatX)*batch_var
                self.bn_updates = [(self.avg_batch_mean, new_m), (self.avg_batch_var, new_v)]

        if hasattr(self, 'g'):
            activation = norm_features*self.g.dimshuffle(*self.dimshuffle_args)
        else:
            activation = norm_features
        if hasattr(self, 'b'):
            activation += self.b.dimshuffle(*self.dimshuffle_args)

        return self.nonlinearity(activation)

def batch_norm(layer, b=lasagne.init.Constant(0.), g=lasagne.init.Constant(1.), **kwargs):
    """
    adapted from https://gist.github.com/f0k/f1a6bd3c8585c400c190
    """
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = lasagne.nonlinearities.identity
    else:
        nonlinearity = lasagne.nonlinearities.identity
    if hasattr(layer, 'b'):
        del layer.params[layer.b]
        layer.b = None
    return BatchNormLayer(layer, b, g, nonlinearity=nonlinearity, **kwargs)

class GaussianNoiseLayer(lasagne.layers.Layer):
    def __init__(self, incoming, sigma=0.1, **kwargs):
        super(GaussianNoiseLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        self.sigma = sigma

    def get_output_for(self, input, deterministic=False, use_last_noise=False, **kwargs):
        if deterministic or self.sigma == 0:
            return input
        else:
            if not use_last_noise:
                self.noise = self._srng.normal(input.shape, avg=0.0, std=self.sigma)
            return input + self.noise


# /////////// older code used for MNIST ////////////

# weight normalization
def l2normalize(layer, train_scale=True):
    W_param = layer.W
    s = W_param.get_value().shape
    if len(s)==4:
        axes_to_sum = (1,2,3)
        dimshuffle_args = [0,'x','x','x']
        k = s[0]
    else:
        axes_to_sum = 0
        dimshuffle_args = ['x',0]
        k = s[1]
    layer.W_scale = layer.add_param(lasagne.init.Constant(1.),
                          (k,), name="W_scale", trainable=train_scale, regularizable=False)
    layer.W = W_param * (layer.W_scale/T.sqrt(1e-6 + T.sum(T.square(W_param),axis=axes_to_sum))).dimshuffle(*dimshuffle_args)
    return layer

# fully connected layer with weight normalization
class DenseLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_units, theta=lasagne.init.Normal(0.1), b=lasagne.init.Constant(0.),
                 weight_scale=lasagne.init.Constant(1.), train_scale=False, nonlinearity=relu, **kwargs):
        super(DenseLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (lasagne.nonlinearities.identity if nonlinearity is None else nonlinearity)
        self.num_units = num_units
        num_inputs = int(np.prod(self.input_shape[1:]))
        self.theta = self.add_param(theta, (num_inputs, num_units), name="theta")
        self.weight_scale = self.add_param(weight_scale, (num_units,), name="weight_scale", trainable=train_scale)
        self.W = self.theta * (self.weight_scale/T.sqrt(T.sum(T.square(self.theta),axis=0))).dimshuffle('x',0)
        self.b = self.add_param(b, (num_units,), name="b")

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, init=False, deterministic=False, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        activation = T.dot(input, self.W)

        if init:
            ma = T.mean(activation, axis=0)
            activation -= ma.dimshuffle('x',0)
            stdv = T.sqrt(T.mean(T.square(activation),axis=0))
            activation /= stdv.dimshuffle('x',0)
            self.init_updates = [(self.weight_scale, self.weight_scale/stdv), (self.b, -ma/stdv)]
        else:
            activation += self.b.dimshuffle('x', 0)

        return self.nonlinearity(activation)


# comes from Ishamel code base
def conv_layer(input_, filter_size, num_filters, stride, pad, nonlinearity=relu, W=Normal(0.02), **kwargs):
    return layers.conv.Conv2DDNNLayer(input_,
                                      num_filters=num_filters,
                                      stride=parse_tuple(stride),
                                      filter_size=parse_tuple(filter_size),
                                      pad=pad,
                                      W=W, nonlinearity=nonlinearity, **kwargs)


class BilinearUpsampling(lasagne.layers.Layer):
    def __init__(self, input_, ratio, use_1D_kernel=True, **kwargs):
        super(BilinearUpsampling, self).__init__(input_, **kwargs)
        self.ratio = ratio
        self.use_1D_kernel = use_1D_kernel

    def get_output_shape_for(self, input_shape):
        dims = input_shape[2:]
        output_dims = tuple(c * self.ratio for c in dims)
        return input_shape[:2] + output_dims

    def get_output_for(self, input_, **kwargs):
        return bilinear_upsampling(input_, ratio=self.ratio,
                                   batch_size=self.input_shape[0],
                                   num_input_channels=self.input_shape[1],
                                   use_1D_kernel=self.use_1D_kernel)


def resnet_block(input_, filter_size, num_filters,
                 activation=relu, downsample=False,
                 no_output_act=True,
                 use_shortcut=False,
                 use_wn=False,
                 W_init=Normal(0.02),
                 **kwargs):
    """
    Resnet block layer.
    """

    normalization = weight_norm if use_wn else batch_norm

    block = []
    _stride = 2 if downsample else 1
    # conv -> BN -> Relu
    block.append(normalization(conv_layer(input_, filter_size, num_filters,
                                       _stride, 'same', nonlinearity=activation,
                                       W=W_init
    )))
    # Conv -> BN
    block.append(normalization(conv_layer(block[-1], filter_size, num_filters, 1, 'same', nonlinearity=None,
                                       W=W_init)))

    if downsample or use_shortcut:
        shortcut = conv_layer(input_, 1, num_filters, _stride, 'valid', nonlinearity=None)
        block.append(ElemwiseSumLayer([shortcut, block[-1]]))
    else:
        block.append(ElemwiseSumLayer([input_, block[-1]]))

    if not no_output_act:
        block.append(NonlinearityLayer(block[-1], nonlinearity=activation))

    return block[-1]
