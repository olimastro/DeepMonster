import numpy as np
import theano
import theano.tensor as T
from initializations import Constant


def weight_norm(layer, train_g=None):
    """
        Applies weight norm on a layer
        train_g None := no g at all
        train_g False := g gets a value but is not propagated on as a param
        train_g True := same as False but it is updated
    """
    assert train_g in [None, False, True]
    init_g = Constant(1.)

    try:
        weight_tag = 'W' if hasattr(layer, 'W') else 'U'
    except AttributeError:
        raise AttributeError("Trying to call weight norm on {} ".format(layer)+\
                             "without layer.W or layer.U defined")
    weights = getattr(layer, weight_tag)

    Wndim = weights.get_value().ndim
    if Wndim == 4:
        W_axes_to_sum = (1,2,3)
        W_dimshuffle_args = (0,'x','x','x')
    elif Wndim == 5:
        W_axes_to_sum = (1,2,3,4)
        W_dimshuffle_args = (0,'x','x','x','x')
    # a bit sketch but serves our purpose for the LSTM weights
    #elif weight_tag == 'U' and Wndim == 2:
    #    W_axes_to_sum = 1
    #    W_dimshuffle_args = (0,'x')
    elif Wndim == 3 :
        raise NotImplementedError("What is a weight with 3 dimensions?")
    else :
        W_axes_to_sum = 0
        W_dimshuffle_args = ('x',0)

    if train_g is not None:
        g = init_g(layer.output_dims)
        g = theano.shared(g, name=layer.prefix+'_g')
        if train_g :
            layer.params += [g]

        new_weights = weights * (
             g / T.sqrt(1e-6 + T.sum(T.square(weights),
                                     axis=W_axes_to_sum))).dimshuffle(*W_dimshuffle_args)
        layer.g = g
    else:
        new_weights = weights / \
                T.sqrt(1e-6 + T.sum(T.square(weights),
                                    axis=W_axes_to_sum,keepdims=True))

    setattr(layer, weight_tag, new_weights)



def batch_norm(x, betas, gammas, mean=None, var=None,
               mean_only=False, axis='auto', eps=1e-5):
    assert (mean is None and var is None) or \
            (not mean is None and not var is None)
    if axis == 'auto':
        if x.ndim == 2:
            axis = (0,)
        elif x.ndim == 4 :
            axis = (0,2,3,)
        else:
            raise ValueError("Dims {} in batch norm?".format(x.ndim))
    pattern = []
    j = 0
    for i in range(x.ndim):
        if i in axis:
            pattern.append('x')
        else:
            pattern.append(j)
            j += 1

    bn_mean = x.mean(axis=axis, keepdims=True)
    def parse_bg(v):
        # useless mean but we need to know its pattern for parsing
        if isinstance(v, (int, float)):
            return v * T.ones_like(bn_mean)
        return v.dimshuffle(*pattern)

    betas = parse_bg(betas)
    gammas = parse_bg(gammas)

    if not mean_only:
        bn_var = T.mean(T.sqr(x - bn_mean), axis=axis, keepdims=True)
    else:
        bn_var = T.ones_like(bn_mean)
    bn_var += np.float32(eps)

    def apply(x, mean, var):
        return gammas * ((x - mean) / var) + betas

    if mean is None:
        rx = apply(x, bn_mean, bn_var)
        rm = bn_mean
        rv = bn_var
    else:
        rx = apply(x, mean, var)
        rm = None
        rv = None

    return rx, rm, rv
