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



def batch_norm(x, betas, gammas, mean=None, invstd=None, axes='auto'):
    assert (mean is None and invstd is None) or \
            (not mean is None and not invstd is None)
    if axes == 'auto':
        if x.ndim == 2:
            axes = 'per-activation'
        elif x.ndim == 4 :
            axes = 'spatial'
        else:
            raise ValueError("Dims {} in batch norm?".format(x.ndim))
    if mean is None:
        rx, rm, rv = T.nnet.bn.batch_normalization_train(x, gammas, betas, axes=axes)
    else:
        var = T.sqr(1. / invstd)
        rm = None
        rv = None
        rx = T.nnet.bn.batch_normalization_test(x, gammas, betas, mean, var, axes=axes)

    return rx, rm, rv
