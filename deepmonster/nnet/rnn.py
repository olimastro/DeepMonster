import inspect

import utils
from activations import Identity
from baselayers import AbsLayer
from convolution import ConvLayer, DeConvLayer
from scanlayers import ScanConvLSTM, ScanLSTM
from simple import FullyConnectedLayer, ZeroLayer


class RecurrentLayer(AbsLayer):
    """
        A reccurent layer consists of two applications of somewhat independant
        layers: one that does an application like a feedforward, and the other
        that does the application through time.

        A rnn can also be used in mostly two fashion. It is either fed its own
        output for the next time step or it computes a whole sequence. In case
        1), we only need one theano scan which is outside what is actually just
        a normal FeedForwardNetwork. In case 2), every single instance of an
        rnn needs to have its own theano scan.

        This class, with ScanLayer class, is intended to handle all these cases.
        NOTE: It should be possible to use a non scanlayer for the time application,
        in this  case if no step is implemented, this class will call the fprop
        of that layer.

        NEW: Can now use without an upwardlayer defined.
    """
    def __init__(self, scanlayer, upwardlayer=None, mode='auto', time_collapse=True):
        assert mode in ['scan', 'out2in', 'auto']
        self.mode = mode
        self.scanlayer = scanlayer
        self.upwardlayer = ZeroLayer(scanlayer.spatial_input_dims) \
                if upwardlayer is None else upwardlayer
        self.time_collapse = False \
                if isinstance(upwardlayer, ZeroLayer) else time_collapse

    @property
    def prefix(self):
        return self._prefix

    @prefix.setter
    def prefix(self, value):
        self.upwardlayer.prefix = value
        self.scanlayer.prefix = value
        self._prefix = value

    @property
    def params(self):
        return self.upwardlayer.params + self.scanlayer.params

    @property
    def input_dims(self):
        return self.upwardlayer.input_dims

    @property
    def output_dims(self):
        return self.scanlayer.output_dims

    @property
    def accepted_kwargs_fprop(self):
        kwargs = super(RecurrentLayer, self).accepted_kwargs_fprop
        kwargs.update(self.scanlayer.accepted_kwargs_fprop)
        kwargs.update(self.upwardlayer.accepted_kwargs_fprop)
        return kwargs

    @property
    def outputs_info(self):
        return self.scanlayer.outputs_info


    def get_outputs_info(self, *args):
        return self.scanlayer.get_outputs_info(*args)


    def set_attributes(self, attributes):
        self.upwardlayer.set_attributes(attributes)
        self.scanlayer.set_attributes(attributes)


    def initialize(self, dims):
        self.upwardlayer.initialize(dims)
        self.scanlayer.initialize(self.upwardlayer.output_dims)


    def set_io_dims(self, tup):
        print "----------------> hallo? <-----------------"
        self.upwardlayer.set_io_dims(tup)
        self.scanlayer.set_io_dims(self.upwardlayer.output_dims)


    def fprop(self, x=None, outputs_info=None, **kwargs):
        """
            This fprop should deal with various setups. if x.ndim == 5 it is
            pretty easy, every individual fprop of the rnn should handle this
            case easily since the fprop through time has its own time
            implementation.

            if x.ndim == 4 now it gets funky. Are we inside a for loop or a
            inside a theano scan?
            Since a for loop is easier, lets consider the scan case and the for
            loop user shall adapt. In this case kwargs should contain outputs_info
            which IN THE SAME ORDER should correspond
            to the reccurent state that the scanlayer.step is using.
        """
        if x is None or isinstance(self.upwardlayer, ZeroLayer):
            in_up = None
            mode = 'out2in'
        else:
            # logic here is that if x.ndim is 2 or 4, x is in bc or bc01
            # for 3 or 5 x is in tbc or tbc01. When t is here, you want to
            # scan on the whole thing.
            if self.mode == 'auto':
                mode = 'scan' if x.ndim in [3, 5] else 'out2in'
            else:
                mode = self.mode

            if x.ndim in [2, 4]:
                assert mode == 'out2in'
            if self.time_collapse and mode == 'scan':
                # collapse batch and time together
                in_up, xshp = utils.collapse_time_on_batch(x)
            else:
                in_up = x

        # forward pass
        h = self.upwardlayer.fprop(in_up, **kwargs)

        # sketchy but not sure how to workaround
        # scan step function doesnt accept keywords
        self.scanlayer.deterministic = kwargs.pop('deterministic', False)

        if mode == 'out2in':
            if not hasattr(self.scanlayer, 'step'):
                # hmm maybe this can work?
                return self.scanlayer.fprop(h)

            # the outputs_info of the outside scan should contain the reccurent state
            if outputs_info is None:
                raise RuntimeError(
                    "There should be an outputs_info in fprop of "+self.prefix)

            # parse the format correctly
            outputs_info = list(outputs_info) if (isinstance(outputs_info, list)\
                    or isinstance(outputs_info, tuple)) else [outputs_info]

            # this calls modify outputs info in the dict, but it should be fine
            self.scanlayer.before_scan(h, axis=0, outputs_info=outputs_info)
            args = tuple(self.scanlayer.scan_namespace['sequences'] + \
                         self.scanlayer.scan_namespace['outputs_info'] + \
                         self.scanlayer.scan_namespace['non_sequences'])
            scanout = self.scanlayer.step(*args)
            y = self.scanlayer.after_scan(scanout[0], scanout[1])

        elif mode == 'scan':
            kwargs.update({'outputs_info': outputs_info})
            if self.time_collapse:
                # reshape to org tensor ndim
                h = utils.expand_time_from_batch(h, xshp)
            y = self.scanlayer.apply(h, **kwargs)

        return y


#TODO: Intercept Initilization keyword properly
# All the classes below are intended to add sugar and define default behaviors.
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
        if isinstance(self.upwardlayer, ZeroLayer):
            self.scanlayer.batch_norm_on_x = False
        self.scanlayer.batch_norm = False


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
    def __init__(self, output_dims, input_dims=None, upward='default', time='default', **kwargs):
        output_dims = utils.parse_tuple(output_dims)

        scan_spatial_input_dims = (output_dims[0]*4,)+output_dims[1:]
        if upward == 'default':
            upward = FullyConnectedLayer(output_dims=scan_spatial_input_dims,
                                         input_dims=input_dims, **kwargs)

        # there is no kwargs proper to a fully
        #kwargs = self.popkwargs(upward, kwargs)
        if time =='default':
            time = ScanLSTM(output_dims=output_dims, input_dims=output_dims,
                            spatial_input_dims=scan_spatial_input_dims, **kwargs)

        super(LSTM, self).__init__(time, upward, **kwargs)



class ConvLSTM(TypicalReccurentLayer):
    """
        Generic ConvLSTM class

        REMINDER: Take care with those * 4
    """
    def __init__(self, filter_size, num_filters,
                 time_filter_size=None, time_num_filters=None,
                 convupward='conv', convtime='conv', **kwargs):
        if time_filter_size is None:
            time_filter_size = utils.parse_tuple(filter_size, 2)
        else:
            time_filter_size = utils.parse_tuple(time_filter_size, 2)

        # the time application doesnt change the dimensions
        # it is achieved through filter_size of odd shape with half padding
        assert time_filter_size[0] % 2 == 1
        if time_num_filters is None:
            time_num_filters = num_filters

        scan_spatial_input_dims = num_filters * 4

        if convupward == 'conv':
            convupward = ConvLayer(filter_size, scan_spatial_input_dims, **kwargs)
        elif convupward == 'deconv':
            convupward = DeConvLayer(filter_size, scan_spatial_input_dims, **kwargs)

        kwargs = self.popkwargs(convupward, kwargs)
        if convtime == 'conv':
            convtime = ScanConvLSTM(time_filter_size, time_num_filters,
                                    num_channels=num_filters,
                                    spatial_input_dims=scan_spatial_input_dims, **kwargs)

        super(ConvLSTM, self).__init__(convtime, convupward, **kwargs)



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
