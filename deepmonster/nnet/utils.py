import inspect
import numpy as np

import theano
import theano.tensor as T

def getftensor5():
    return T.TensorType('float32', (False,)*5)


def getnumpyf32(size):
    return np.random.random(size).astype(np.float32)


def infer_odim_conv(i, k, s):
    return (i-k) // s + 1


def infer_odim_convtrans(i, k, s) :
    return s*(i-1) + k


def log_sum_exp(x, axis=1):
    m = T.max(x, axis=axis)
    return m+T.log(T.sum(T.exp(x-m.dimshuffle(0,'x')), axis=axis))


# http://kbyanc.blogspot.ca/2007/07/python-aggregating-function-arguments.html
def find_func_kwargs(func):
    argspec = inspect.getargspec(func)
    if argspec.defaults is not None:
        return argspec.args[-len(argspec.defaults):]
    else:
        return []


def collapse_time_on_batch(x):
    xshp = tuple([x.shape[i] for i in range(x.ndim)])
    y = x.reshape((xshp[0] * xshp[1],) + xshp[2:])
    return y, xshp

def expand_time_from_batch(x, orgshp):
    xshp = tuple([x.shape[i] for i in range(x.ndim)])
    return x.reshape((orgshp[0], orgshp[1],) + xshp[1:])

def stack_time(x, y):
    assert y.ndim == x.ndim + 1, "Cannot stack tensors without the same dimensions after time dim"
    tup = ('x',) + tuple(range(x.ndim))
    return T.concatenate([x.dimshuffle(*tup), y], axis=0)


def prefix_vars_list(varlist, prefix):
    prefix = prefix + '_'
    return [prefix + var.name for var in varlist]


def parse_tuple(tup, length=1) :
    if isinstance(tup, tuple):
        return tup
    return (tup,) * length


def get_gradients_list(feedforwards, y):
    """
        Helper function to get a list of all the gradients of y with respect to
        the output of each layer.

        Use in conjonction with fprop_passes of Feedforward while doing fprops
    """
    grads = []
    for feedforward in feedforwards:
        for fprop_pass in feedforward.fprop_passes.keys():
            # activations_list[0] is the input of the block, discard
            activations_list = feedforward.fprop_passes[fprop_pass]
            for layer, activation in zip(feedforward.layers, activations_list[1:]):
                grad = T.grad(y, activation)
                grad.name = '{}_d{}/d{}'.format(fprop_pass, y.name, layer.prefix)
                grads += [grad]

    return grads


def find_bn_params(anobject):
    """
        Helper function to find the batch norm params. It tries to be as helpful
        as it can so an object can be:
            - a layer object
            - a list of layers
            - a feedforward object
            - a list of feedforward object
            - any combination of above
    """
    # a feedforward is characterized by its 'layers' attribute
    layers = []
    if isinstance(anobject, list):
        for x in anobject:
            if hasattr(x, 'layers'):
                layers += x.layers
            else:
                layers += x
    elif hasattr(anobject, 'layers'):
        layers += anobject.layers
    else:
        layers = [anobject]

    updt = []
    for layer in layers:
        # RNN batch norm is a MESS
        if hasattr(layer, '_updates'):
            updt.extend(layer._updates)
        elif hasattr(layer, 'bn_updates'):
            updt.extend(layer.bn_updates)
    return updt
