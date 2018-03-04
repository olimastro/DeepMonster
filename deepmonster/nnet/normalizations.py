import numpy as np
import theano
import theano.tensor as T

from baselayers import ParametrizedLayer
from initializations import Constant
from utils import getfloatX, get_dm_axis_info
from deepmonster.utils import make_dict_cls_name_obj


def batch_norm(x, betas, gammas, mean=None, std=None,
               cbn=False, mean_only=False, axis='auto', eps=1e-4):
    eps = getfloatX()(eps)
    assert (mean is None and std is None) or \
            (not mean is None and not std is None)
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

    # so hacky
    if cbn:
        betas = betas.dimshuffle(0, 1, 'x', 'x')
        gammas = betas.dimshuffle(0, 1, 'x', 'x')
    else:
        betas = parse_bg(betas)
        gammas = parse_bg(gammas)

    if not mean_only:
        bn_std = T.mean(T.sqr(x - bn_mean), axis=axis, keepdims=True)
    else:
        bn_std = T.ones_like(bn_mean)

    def apply(x, mean, std):
        return (x - mean) * gammas / T.sqrt(std + eps) + betas

    if mean is None:
        rx = apply(x, bn_mean, bn_std)
        rm = bn_mean
        rs = bn_std
    else:
        rx = apply(x, mean, std)
        rm = None
        rs = None

    return rx, rm, rs



class ActivationNormLayer(ParametrizedLayer):
    str_pattern = NotImplemented

    def __init__(self, mean_only=False, eps=1e-4, **kwargs):
        super(ActivationNormLayer, self).__init__(**kwargs)
        self.mean_only = mean_only
        self.eps = getfloatX()(eps)


    @property
    def param_dict_initialization(self):
        param_dict = {
            'betas' : [self.output_dims[0], 'zeros'],
            'gammas' : [self.output_dims[0], 'ones'],
        }
        return param_dict


    def convert_str_pattern(self, ndim):
        axis_info = get_dm_axis_info()[ndim]
        try:
            axis = [axis_info.index(i) for i in self.str_pattern]
        except ValueError:
            raise ValueError("Could not find requested axis {} for an input of ndim {}".format(
                self.str_pattern, ndim))
        return axis


    def apply(self, x):
        axis = self.convert_str_pattern(x.ndim)
        mean = x.mean(axis=axis, keepdims=True)

        if not self.mean_only:
            std = T.std(x, axis=axis, keepdims=True)
        else:
            std = T.ones_like(bn_mean)

        pattern = list(('x',) * x.ndim)
        pattern[get_dm_axis_info()[x.ndim].index('c')] = 0

        betas = self.betas.dimshuffle(*pattern)
        gammas = self.gammas.dimshuffle(*pattern)

        rval = (x - mean) * gammas / T.sqrt(std + self.eps) + betas
        #self.tag_bn_vars(mean, 'mean' + key + self.prefix)
        #self.tag_bn_vars(std, 'std' + key + self.prefix)
        return rval


class BatchNorm(ActivationNormLayer):
    str_pattern = ('b','c')

class SpatialBatchNorm(ActivationNormLayer):
    str_pattern = ('b','c','0','1')

class LayerNorm(ActivationNormLayer):
    str_pattern = ('c',)

class InstanceNorm(ActivationNormLayer):
    str_pattern = ('0','1')


    ## -------- Normalization related functions -------------- #
    #### batch norm ###
    #def batch_norm_addparams(self):
    #        self.param_dict.update({
    #            'gammas' : [self.output_dims[0], 'ones']
    #        })


    #def tag_bn_vars(self, var, name):
    #    # tag the graph so popstats can use it
    #    var.name = name
    #    setattr(var.tag, 'bn_statistic', name)


    #def bn(self, x, betas=None, gammas=None, key='_', deterministic=False, axis='auto'):
    #    """
    #        BN is the king of pain in the ass especially in the case of RNN
    #        (which is actually why this whole library was written for at first).

    #        BN is to be used with get_inference_graph in popstats at inference phase.
    #        It will compute the batch statistics from a dataset and replace in the
    #        theano graph the tagged bnstat with the computed values.

    #        All the deterministic logic is therefore deprecated here.
    #    """
    #    # make sure the format of the key is _something_
    #    if key != '_':
    #        if '_' != key[0]:
    #            key = '_' + key
    #        if '_' != key[-1]:
    #            key = key + '_'

    #    #if deterministic:
    #    #    print "WARNING: deterministic=True is deprecated in Layer.bn and has"+\
    #    #            " no effect"

    #    mean, std = (None, None,)

    #    if betas is None:
    #        betas = getattr(self, 'betas', 0.)
    #    if gammas is None:
    #        gammas = getattr(self, 'gammas', 1.)
    #    rval, mean, std = batch_norm(x, betas, gammas, mean, std,
    #                                 cbn=self.conditional_batch_norm,
    #                                 mean_only=self.bn_mean_only,
    #                                 axis=axis)

    #    # do not tag on cbn
    #    if not deterministic and not self.conditional_batch_norm:
    #        self.tag_bn_vars(mean, 'mean' + key + self.prefix)
    #        self.tag_bn_vars(std, 'std' + key + self.prefix)

    #    return rval
    ####

    #### weight norm ###
    ##FIXME: the dimshuffle on the mean and var depends on their dim.
    ## Easy for 2&4D, but for a 5D or 3D tensor?
    #def init_wn(self, x, init_stdv=0.1):
    #    raise NotImplementedError("You can use init_wn for now by doing batch +\
    #                              norm on first layer")
    #    m = T.mean(x, self.wn_axes_to_sum)
    #    x -= m.dimshuffle(*self.wn_dimshuffle_args)
    #    inv_stdv = init_stdv/T.sqrt(T.mean(T.square(x), self.wn_axes_to_sum))
    #    x *= inv_stdv.dimshuffle(*self.wn_dimshuffle_args)
    #    self.wn_updates = [(self.betas, -m*inv_stdv), (self.g, self.g*inv_stdv)]

    #    return x
    # ------------------------------------------------------- #
