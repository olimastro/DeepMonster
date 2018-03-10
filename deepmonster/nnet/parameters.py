#from . import tensorlib as T
from initializations import Constant

from deepmonster import config

import theano
import theano.tensor as T

#TODO: param class to get backend agnostic
class Parameters(object):
    if config.backend == 'theano':
        import theano
        backend_param = theano.shared(np_matrix, name=name)
    elif config.backend == 'pytorch':
        import torch
        backend_param = torch.autograd.Variable(np_matrix, requires_grad=True)
    elif config.backend == 'tensorflow':
        raise NotImplementedError
    else:
        # this should really not happen, backends were checked twice already
        import ipdb; ipdb.set_trace()
    __backend = None
    def __init__(self, np_matrix, name=''):
        self.param = self.__backend()
        self.name = name

    def get_value(self):
        pass


class ParametersNormalization(object):
    pass


class WeightNormalization(ParametersNormalization):
    def __init__(self, train_g=None):
        assert train_g in [None, False, True], "invalid train_g value"
        self.train_g = train_g


    def apply_on_layer(self, layer):
        """Applies weight norm on a layer
        - train_g None := no g at all
        - train_g False := g gets a value but is not propagated on as a param
        - train_g True := same as False but it is updated
        """
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
        elif Wndim == 3 :
            raise NotImplementedError("What is a weight with 3 dimensions?")
        else :
            W_axes_to_sum = 0
            W_dimshuffle_args = ('x',0)

        if self.train_g is not None:
            g = init_g(layer.output_dims)
            g = theano.shared(g, name=layer.prefix+'_g')
            if self.train_g :
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
