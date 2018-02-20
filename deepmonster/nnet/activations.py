import numpy as np
import theano
import theano.tensor as T

class Activation(object):
    """
        An activation has to redifine __call__,
        everything else is optional

        This class template is used for if conditions
    """
    def __call__(self):
        raise NotImplemented


class LeakyRectifier(Activation) :
    def __init__(self, leak=0.1) :
        self.leak = leak

    def __call__(self, input_) :
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

    def __call__(self, input_):
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
    def __call__(self, input_):
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
    def __call__(self, input_) :
        return T.tanh(input_)


class ClipActivation(Activation) :
    def __init__(self, high=1., low=None):
        # if low is none, will take -high
        self.high = high
        if low is None:
            low = -high
        self.low = low
        assert self.high > self.low

    def __call__(self, input_) :
        return T.clip(input_, self.low, self.high)

class HardTanh(ClipActivation):
    def __init__(self, **kwargs):
        self.high = 1.
        self.low = -1.


class Identity(Activation) :
    def __call__(self, input_) :
        return input_


class Sigmoid(Activation) :
    def __call__(self, input_) :
        return T.nnet.sigmoid(input_)


class BinarySigmoid(Sigmoid):
    def __call__(self, input_) :
        rval = super(BinarySigmoid, self).__call__(input_)
        mask = T.ge(rval, 0.5)
        return mask


class Softmax(Activation) :
    def __call__(self, input_) :
        return T.nnet.softmax(input_)


class ChannelSoftmax(Softmax):
    # applies softmax on the channel axis of up to a 5D tensor
    # the channel axis is deepmonster default according to ndim
    def __call__(self, input_):
        if input_.ndim == 2:
            return super(ChannelSoftmax, self).__call__(input_)

        shp = [input_.shape[i] for i in range(input_.ndim)]
        if input_.ndim in [3, 5]:
            # collapse time on batch
            x = input_.reshape((shp[0] * shp[1]) + tuple(shp[2:]))

        if input_.ndim == 3:
            rval = super(ChannelSoftmax, self).__call__(x)
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
            rval = super(ChannelSoftmax, self).__call__(rval)
            # inflate
            rval = rval.reshape((x.shape[0], x.shape[1], x.shape[2]))
            rval = rval.dimshuffle(0, 2, 1)
            rval = rval.reshape((rval.shape[0], shp[-3], shp[-2], shp[-1]))

        if input_.ndim == 5:
            rval = rval.reshape(tuple(shp))

        return rval


class Swish(Activation):
    def __call__(self, input_):
        return input_ * T.nnet.sigmoid(input_)


class GELU(Activation):
    def __call__(self, input_):
        return 0.5 * input_ * (1. + T.tanh(T.sqrt(2. / np.pi) * \
            (input_ + 0.044715 * input_**3)))


class UnitNorm(Activation):
    def __call__(self, input_):
        # normalize as a unit vector the last axis
        assert input_.ndim == 2
        return input_ / T.sqrt(T.sum(input_**2, axis=1, keepdims=True))


if __name__ == '__main__' :
    convmaxout = ConvMaxout(2)
    x = T.ftensor4('x')
    out = convmaxout(x)
    f = theano.function([x],out, allow_input_downcast=True)

    inp = np.random.random((218,32,28,28))
    print f(inp).shape
