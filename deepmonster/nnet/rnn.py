import inspect

import utils
from activations import Identity
from baselayers import EmptyLayer, RecurrentLayer
from convolution import ConvLayer, DeConvLayer
from scanlayers import ScanConvLSTM, ScanLSTM
from simple import FullyConnectedLayer

# This file contains all the helper class for RecurrentLayer. It is possible to
# create a custom RecurrentLayer only by using the RecurrentLayer class. All the
# classes here are intended to add sugar and define default behaviors.

#TODO: Intercept Initilization keyword properly

class TypicalReccurentLayer(RecurrentLayer):
    """
        A typical rnn doesn't have two biases applications and applies batch norm
        only in the time computation phase. Add more as it goes!

        Don't worry about kwargs having things like batch_norm=True, it won't conflict
        with RecurrentLayer as this class dosen't pass kwargs to RecurrentLayer.
    """
    def __init__(self, *args, **kwargs):
        mode = kwargs.pop('mode', 'auto')
        time_collapse = kwargs.pop('time_collapse', True)
        super(TypicalReccurentLayer, self).__init__(*args, mode=mode, time_collapse=time_collapse)
        self.upwardlayer.use_bias = False
        self.upwardlayer.batch_norm = False
        self.upwardlayer.activation = Identity()
        if isinstance(self.upwardlayer, EmptyLayer):
            self.scanlayer.batch_norm_on_x = False


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


# generic LSTMs don't change dimensions in the reccurence
class LSTM(TypicalReccurentLayer):
    """
        Generic LSTM class

        REMINDER: Take care with those * 4
    """
    def __init__(self, output_dims, input_dims=None, upward=None, time=None, **kwargs):
        output_dims = utils.parse_tuple(output_dims)

        if upward is None:
            upward = FullyConnectedLayer(output_dims=(output_dims[0]*4,)+output_dims[1:],
                                         input_dims=input_dims, **kwargs)

        # there is no kwargs proper to a fully
        #kwargs = self.popkwargs(upward, kwargs)
        if time is None:
            time = ScanLSTM(output_dims=output_dims, input_dims=output_dims, **kwargs)

        super(LSTM, self).__init__(upward, time, **kwargs)



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
        else:
            time_filter_size = utils.parse_tuple(time_filter_size, 2)

        # the time application doesnt change the dimensions
        # it is achieved through filter_size of odd shape with half padding
        assert time_filter_size[0] % 2 == 1
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
    from network import Feedforward
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
    xnp = np.random.random((10,50,3,1,1)).astype(np.float32)
    ftensor5 = T.TensorType('float32', (False,)*5)
    x = ftensor5('x')
    x.tag.test_value = xnp

    layers = [
        LSTM(output_dims=100, input_dims=(3,1,1))
    ]

    ff = Feedforward(layers, 'lstm', **config)
    ff.initialize()
    y = ff.fprop(x)
    #cost = y[-1].mean()

    f = theano.function(inputs=[x], outputs=[y])
    out = f(xnp)
    print out[0].shape
