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

    layer.W_param = layer.W
    Wndim = layer.W.get_value().ndim
    if Wndim == 4:
        W_axes_to_sum = (1,2,3)
        W_dimshuffle_args = (0,'x','x','x')
        #else:
        #    W_axes_to_sum = (0,2,3)
        #    W_dimshuffle_args = ('x',0,'x','x')
    elif Wndim == 3 :
        raise NotImplementedError
    else :
        W_axes_to_sum = 0
        W_dimshuffle_args = ('x',0)

    #if len(W_dimshuffle_args) == 2:
    #    layer.wn_axes_to_sum = (0,)
    #else :
    layer.wn_axes_to_sum = (0,2,3)
    layer.wn_dimshuffle_args = ('x',0,'x','x')

    if train_g is not None:
        g = init_g(layer.output_size)
        g = theano.shared(g, name=layer.prefix+'_g')
        if train_g :
            layer.params += [g]

        layer.W = layer.W_param * (
             g / T.sqrt(1e-6 + T.sum(T.square(layer.W_param),
                                     axis=W_axes_to_sum))).dimshuffle(*W_dimshuffle_args)
        layer.g = g
    else:
        layer.W = layer.W_param / \
                T.sqrt(1e-6 + T.sum(T.square(layer.W_param),
                                    axis=W_axes_to_sum,keepdims=True))



def batch_norm(x, betas, gammas, bn_mean_only=False):
    # TODO: make spatial batch_norm optional
    if x.ndim == 2:
        axis = 1
        pattern = ('x',0)
    elif x.ndim == 4 :
        axis = [0, 2, 3] # this implies spatial batch norm
        pattern = ('x',0,'x','x')
    else:
        raise ValueError("Dims {} in batch norm?".format(x.ndim))

    mean = x.mean(axis=axis, keepdims=True)
    if not bn_mean_only :
        var = T.mean(T.sqr(x - mean), axis=axis, keepdims=True)
    else :
        var = theano.tensor.ones_like(mean)

    if betas == 0 :
        pass
    elif betas.ndim == 1:
        betas = betas.dimshuffle(pattern)
    elif betas.ndim == 3:
        betas = betas.dimshuffle((x.ndim-3)*('x',)+(0,1,2,))

    var_corrected = var + 1e-6
    y = theano.tensor.nnet.bn.batch_normalization(
        inputs=x, gamma=gammas.dimshuffle(pattern), beta=betas,
        mean=mean,
        std=theano.tensor.sqrt(var_corrected),
        mode="low_mem")
    return y, mean, var
