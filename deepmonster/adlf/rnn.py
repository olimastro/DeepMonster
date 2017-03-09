import inspect

import utils
from baselayers import RecurrentLayer
from scanlayers import ScanConvLSTM
from convolution import ConvLayer, DeConvLayer

# This file contains all the helper class for RecurrentLayer. It is possible to
# create a custom RecurrentLayer only by using the RecurrentLayer class. All the
# classes here are intended to add sugar and define default behaviors.


class TypicalReccurentLayer(RecurrentLayer):
    """
        A typical rnn doesn't have two biases applications and applies batch norm
        only in the time computation phase. Add more as it goes!
    """
    def __init__(self, *args, **kwargs):
        mode = kwargs.pop('mode', 'auto')
        super(TypicalReccurentLayer, self).__init__(*args, mode=mode)
        self.upwardlayer.use_bias = False
        self.upwardlayer.batch_norm = False
        self.upwardlayer.activation = None


    def popkwargs(self, upwardlayer, kwargs):
        """
            A typical rnn do not want the kwargs SPECIFIC to the upwardlayer
            to go into the constructor of the scanlayer. Everything else
            for the whole rnn / both layers, will go through.
        """
        if not hasattr(upwardlayer, '__init__'):
            return kwargs
        kwargs_upwardlayer = inspect.getargspec(upwardlayer.__init__)
        for arg in kwargs_upwardlayer.args:
            if kwargs.has_key(arg):
                kwargs.pop(arg)
        return kwargs



class ConvLSTM(TypicalReccurentLayer):
    """
        Generic ConvLSTM class

        REMINDER: Take care with those * 4
    """
    def __init__(self, filter_size, num_filters,
                 time_filter_size=None, time_num_filters=None,
                 convupward=None, convtime=None, **kwargs):
        if time_filter_size is None:
            time_filter_size = utils.parse_tuple(filter_size, 2)
        if time_num_filters is None:
            time_num_filters = num_filters

        if convupward is None or convupward is 'conv':
            convupward = ConvLayer(filter_size, num_filters*4, **kwargs)
        elif convupward is 'deconv':
            convupward = DeConvLayer(filter_size, num_filters*4, **kwargs)

        kwargs = self.popkwargs(convupward, kwargs)
        if convtime is None or convtime is 'conv':
            convtime = ScanConvLSTM(time_filter_size, num_filters,
                                    num_channels=num_filters, **kwargs)

        super(ConvLSTM, self).__init__(convupward, convtime, **kwargs)



if __name__ == '__main__':
    import theano
    import theano.tensor as T
    import numpy as np
    from base import Feedforward
    from activations import LeakyRectifier
    from initializations import Initialization, Gaussian

    config = {
        'batch_norm' : True,
        'use_bias' : True,
        'gamma_scale' : 1.,
        'activation' : LeakyRectifier(leak=0.4),
        'initialization' : Initialization({'W' : Gaussian(std=0.05)}),
    }

    theano.config.compute_test_value = 'warn'
    xnp = np.random.random((10,50,3,11,11)).astype(np.float32)
    ftensor5 = T.TensorType('float32', (False,)*5)
    x = ftensor5('x')
    x.tag.test_value = xnp

    layers = [
        ConvLSTM((5,5), 16, image_size=(11,11), num_channels=3),
        #ConvLSTM((3,3), 32),
    ]

    ff = Feedforward(layers, 'convlstm', **config)
    ff.initialize()
    y = ff.fprop(x)
    cost = y[-1].mean()
    import ipdb; ipdb.set_trace()
    grads = T.grad(cost, ff.params)

    f = theano.function(inputs=[x], outputs=[cost])
    out = f(xnp)
    print out[0].shape
