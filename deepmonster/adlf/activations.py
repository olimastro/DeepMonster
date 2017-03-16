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
        pass


class LeakyRectifier(Activation) :
    def __init__(self, leak=0.1) :
        self.leak = leak

    def __call__(self, input_) :
        #return T.nnet.relu(input_, alpha=self.leak)
        return T.maximum(input_, self.leak*input_)


class Rectifier(Activation) :
    def __call__(self, input_) :
        return T.nnet.relu(input_)


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


class HardTanh(Activation):
    def __call__(self, input_) :
        return T.clip(input_, -1., 1.)


class Identity(Activation) :
    def __call__(self, input_) :
        return input_


class Sigmoid(Activation) :
    def __call__(self, input_) :
        return T.nnet.sigmoid(input_)


class Softmax(Activation) :
    def __call__(self, input_) :
        return T.nnet.softmax(input_)


if __name__ == '__main__' :
    convmaxout = ConvMaxout(2)
    x = T.ftensor4('x')
    out = convmaxout(x)
    f = theano.function([x],out, allow_input_downcast=True)

    inp = np.random.random((218,32,28,28))
    print f(inp).shape
