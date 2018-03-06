import theano
import theano.tensor as T

from baselayers import AbsLayer, ParametrizedLayer
from utils import getfloatX, get_dm_axis_info

class Activation(AbsLayer):
    """An activation has to redifine apply, everything else is optional.

    It also redirects __call__ to apply, imitating function calls as what an activation can be seen.
    """
    def __call__(self, x):
        return self.apply(x)


class LeakyRectifier(Activation) :
    def __init__(self, leak=0.1) :
        self.leak = leak

    def apply(self, input_) :
        return T.nnet.relu(input_, alpha=self.leak)

class Rectifier(LeakyRectifier) :
    def __init__(self):
        super(Rectifier, self).__init__(0.)


class ConvMaxout(Activation):
    def __init__(self, num_pieces):
        self.num_pieces = num_pieces

    def get_outdim(self, dims):
        self.num_channels_out = dims[0] // self.num_pieces
        return (self.num_channels_out,) + dims[1:]

    def apply(self, input_):
        if not hasattr(self, 'num_channels_out'):
            print "WARNING: This ConvMaxout did not have num_channels_out set before hand, inferring from the input"
            num_channels_out = input_.shape[1] // self.num_pieces
        else :
            num_channels_out = self.num_channels_out
        input_ = input_.dimshuffle(0, 2, 3, 1)
        new_shape = ([input_.shape[i] for i in range(input_.ndim - 1)] +
                     [num_channels_out, self.num_pieces])
        output = T.max(input_.reshape(new_shape, ndim=input_.ndim + 1),
                       axis=input_.ndim)
        return output.dimshuffle(0, 3, 1, 2)


class ConvMaxout5D(ConvMaxout):
    def apply(self, input_):
        if not hasattr(self, 'num_channels_out'):
            print "WARNING: This ConvMaxout did not have num_channels_out set before hand, inferring from the input"
            num_channels_out = input_.shape[1] // self.num_pieces
        else :
            num_channels_out = self.num_channels_out
        input_ = input_.dimshuffle(0, 1, 3, 4, 2)
        new_shape = ([input_.shape[i] for i in range(input_.ndim - 1)] +
                     [num_channels_out, self.num_pieces])
        output = T.max(input_.reshape(new_shape, ndim=input_.ndim + 1),
                            axis=input_.ndim)
        return output.dimshuffle(0, 1, 4, 2, 3)


class Tanh(Activation) :
    def apply(self, input_) :
        return T.tanh(input_)


class ClipActivation(Activation) :
    def __init__(self, high=1., low=None):
        # if low is none, will take -high
        self.high = high
        if low is None:
            low = -high
        self.low = low
        assert self.high > self.low

    def apply(self, input_) :
        return T.clip(input_, self.low, self.high)

class HardTanh(ClipActivation):
    def __init__(self, **kwargs):
        self.high = 1.
        self.low = -1.


class Identity(Activation) :
    def apply(self, input_) :
        return input_


class Sigmoid(Activation) :
    def apply(self, input_) :
        return T.nnet.sigmoid(input_)


class BinarySigmoid(Sigmoid):
    def apply(self, input_) :
        rval = super(BinarySigmoid, self).apply(input_)
        mask = T.ge(rval, 0.5)
        return mask


class Softmax(Activation) :
    def apply(self, input_) :
        return T.nnet.softmax(input_)


class ChannelSoftmax(Softmax):
    # applies softmax on the channel axis of up to a 5D tensor
    # the channel axis is deepmonster default according to ndim
    def apply(self, input_):
        if input_.ndim == 2:
            return super(ChannelSoftmax, self).apply(input_)

        shp = [input_.shape[i] for i in range(input_.ndim)]
        if input_.ndim in [3, 5]:
            # collapse time on batch
            x = input_.reshape((shp[0] * shp[1]) + tuple(shp[2:]))

        if input_.ndim == 3:
            rval = super(ChannelSoftmax, self).apply(x)
            # inflate
            rval = rval.reshape(tuple(shp))
        else:
            # collapse 01 together
            if input_.ndim == 4:
                x = input_
            x = x.reshape((x.shape[0], shp[-3], shp[-2] * shp[-1]))
            x = x.dimshuffle(0, 2, 1)
            # collapse 01 on batch
            rval = x.reshape((x.shape[0] * x.shape[1], x.shape[2]))
            rval = super(ChannelSoftmax, self).apply(rval)
            # inflate
            rval = rval.reshape((x.shape[0], x.shape[1], x.shape[2]))
            rval = rval.dimshuffle(0, 2, 1)
            rval = rval.reshape((rval.shape[0], shp[-3], shp[-2], shp[-1]))

        if input_.ndim == 5:
            rval = rval.reshape(tuple(shp))

        return rval


class Swish(Activation):
    def apply(self, input_):
        return input_ * T.nnet.sigmoid(input_)


class GELU(Activation):
    def apply(self, input_):
        return 0.5 * input_ * (1. + T.tanh(T.sqrt(2. / np.pi) * \
            (input_ + 0.044715 * input_**3)))


class UnitNorm(Activation):
    def apply(self, input_):
        # normalize as a unit vector the last axis
        assert input_.ndim == 2
        return input_ / T.sqrt(T.sum(input_**2, axis=1, keepdims=True))


## ------------------- Activation normalization related functions ---------------------- #
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
        if hasattr(self, 'tag_norm_vars'):
            self.tag_norm_vars(mean, 'mean')
            self.tag_norm_vars(std, 'std')
        return rval


class BatchNorm(ActivationNormLayer):
    str_pattern = ('b','c')
    def tag_norm_vars(self, var, name):
        name = self.prefix + '_' + name
        var.name = name
        var.tag.bn_statistic = name


class SpatialBatchNorm(BatchNorm):
    str_pattern = ('b','c','0','1')

class LayerNorm(ActivationNormLayer):
    str_pattern = ('c',)

class InstanceNorm(ActivationNormLayer):
    str_pattern = ('0','1')


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
