import inspect
import numpy as np
import theano
import theano.tensor as T

from baselayers import ParametrizedLayer
from convolution import ConvLayer
from simple import FullyConnectedLayer
from utils import parse_tuple


class ScanLayer(ParametrizedLayer):
    """
        General class for layers that has to deal with scan.
        The standard usage of this type of Layer is to work in tandem with RecurrentLayer
        that takes care of coordinating the forward propagation and the time (scan) propagation.
        This base class does not implement a step function for scan.

        In general, a subclass of ScanLayer will have to subclass ScanLayer and
        another class. The LSTM for exemple applies a dot through time, so it
        has the effective application of a fullyconnected. You want the
        information of the fullyconnected for shape propagation.
    """
    # TODO: There is a problem with how the arguments are passed to the step function.
    # If there is another norm that requires params and you want to have it optional with
    # batch norm, having the two sets of keyword is going to crash.
    # NOTE: This is a inherant limitation of scan and step and will probably never be resolved
    def __init__(self, spatial_input_dims=None):
        self.spatial_input_dims = parse_tuple(spatial_input_dims)

    @property
    def non_sequences(self):
        """
            This takes care to infer the arguments to give as non seq to the
            step function.
            The step functions takes as non seq positional arguments
            (such as the weight matrix) and kwargs (such as batch_norm parameters)
        """
        step_arg_list = inspect.getargspec(self.step).args
        param_names = self.param_dict_initialization.keys()

        slice_len = len(param_names)
        sublist = []
        for item in step_arg_list :
            if item in param_names:
                sublist += [item]

        non_seq = []
        for param in sublist :
            non_seq += [getattr(self, param)]

        return non_seq


    # making sure that there is a default value
    @property
    def deterministic(self):
        return getattr(self, '_deterministic', False)

    @deterministic.setter
    def deterministic(self, value):
        assert value is True or value is False
        self._deterministic = value


    # if the rnn is not driven by an input signal, might want
    # to switch the bn application in its implementation in step
    @property
    def batch_norm_on_x(self):
        return getattr(self, '_batch_norm_on_x', True)

    @batch_norm_on_x.setter
    def batch_norm_on_x(self, value):
        assert value is True or value is False
        self._batch_norm_on_x = value


    def fprop(self, forward_input=None, outputs_info=None,
              batch_size=None, time_size=None, strict=True):
        """
            Allow using a ScanLayer outside a RecurrentLayer class (or even inside but without an
            upward layer). There are 4 cases to handle
            which are the combinations of (if there is a forward input and if there is a time input).
            If there is a forward input, fprop simply pipes everything to the apply method.
            If there is not, then it handles these cases
        """
        print "INFO: going through a ScanLayer fprop (bypassing RNN class fprop)"

        #FIX if it ever needs other kwargs in the apply method
        if forward_input is not None:
            return self.apply(forward_input, outputs_info=outputs_info)

        scan_kwargs = {'non_sequences': self.non_sequences,
                       'strict': strict}

        if outputs_info is None:
            if batch_size is None:
                raise RuntimeError("No inputs to infer batch dimension, needs a value for batch_size")
            scan_kwargs.update({'outputs_info': self.get_outputs_info(batch_size)})
        else:
            # outputs_info SHOULD be a list and the batch axis HAS to be the first one
            batch_size = outputs_info[0].shape[1]
            scan_kwargs.update({'outputs_info': outputs_info})

        if forward_input is None:
            if time_size is None:
                raise RuntimeError("No inputs to infer time dimension, needs a value for time_size")
            # The step function is defined with a forward input and we cannot make it optional
            # (and we don't want to have 1000 versions of the step function). The unfortunate
            # solution is to pass zeros as inputs.
            assert hasattr(self, 'spatial_input_dims'), \
                    "Cannot use fprop of a ScanLayer without a spatial input and without defining "+\
                    "a spatial_input_dims. This mode is only usable by passing dummy zeros of the "+\
                    " right dimensions in order to bypass the spatial input."
            x = T.zeros((time_size, batch_size) + self.spatial_input_dims)
            scan_kwargs.update({'sequences': x})
        else:
            scan_kwargs.update({'sequences': forward_input})


        rval, updates = theano.scan(
            self.step,
            **scan_kwargs)
        out = self.after_scan(rval, updates)
        return out


    def step(self):
        """
            Every theano scan has a step!

            *** The arguments in step NEED to have the same name as the parameters,
            meaning if you do a T.dot(z, U) U is linked to self.U at runtime.

            REMINDER: Even if U used in step can be referenced by self.U,
            because of the arcanes of scan, we need to give it as an argument
            to the step function in the list of non_sequences.
            That's mainly why this whole jibber jabber class is for.
        """
        return NotImplemented


    def set_scan_namespace(self, sequences, outputs_info=None, non_sequences=None):
        """
            Every theano scan has a namespace :
                - sequences
                - outputs_info
                - non_sequences (this is most of the time all the shared theanovar)
        """
        if getattr(self, 'scan_namespace', None) is not None:
            # if any wrapping layer hacked self.step, it probably hacked
            # the non_sequences list. We want these changes to be preserved
            non_sequences = self.scan_namespace['non_sequences']
        else:
            non_sequences = self.non_sequences

        if outputs_info is None:
            outputs_info = self.get_outputs_info()
        namespace = {
            'sequences' : sequences,
            'outputs_info' : outputs_info,
            'non_sequences' : non_sequences,
        }
        self.scan_namespace = namespace


    def before_scan(self, sequences, outputs_info=None):
        """Do before scan manipulations and populate the scan namespace.
        """
        self.set_scan_namespace(sequences, outputs_info=outputs_info)


    def scan(self):
        rval, updates = theano.scan(
            self.step,
            sequences=self.scan_namespace['sequences'],
            non_sequences=self.scan_namespace['non_sequences'],
            outputs_info=self.scan_namespace['outputs_info'],
            strict=True)

        return rval, updates


    def after_scan(self, scanout, updates):
        """
            Do manipulations after scan and drop the updates list
        """
        # this is to preserve somewhere all the output a scanlayer might do
        self.outputs_info = tuple(scanout) if (isinstance(scanout, list) \
                or isinstance(scanout, tuple)) else (scanout,)
        return scanout


    def unroll_scan(self):
        rval = [self.scan_namespace['outputs_info']]
        for i in range(self.time_size):
            step_inputs = [s[i] for s in self.scan_namespace['sequences']] + \
                           rval[-1] + self.scan_namespace['non_sequences']
            scan_out = list(self.step(*step_inputs))
            rval += [scan_out]
        # rval is a list of each returned tuple at each time step
        # scan returns a tuple of all elements that have been joined on the time axis
        new_rval = []
        for i in range(len(rval[1])):
            new_rval += [[]]
            for j in range(1, len(rval)):
                new_rval[i] += [rval[j][i]]

        new_new_rval = []
        for i in new_rval:
            new_new_rval += [T.stack(i, axis=0)]

        return tuple(new_new_rval)


    def apply(self, *args, **kwargs):
        """
            Every ScanLayer has the same structure :
                - do things before scan
                - scan
                - do things after scan
        """
        self.before_scan(*args, **kwargs)
        # BROKEN, anyway, never noticed a time gain, only a memory blow up
        if False and self.time_size is not None:
            # if the time_size is specified we can use a for loop, faster++
            rval = self.unroll_scan()
        else:
            rval, updates = self.scan()
        out = self.after_scan(rval, updates)

        return out



class ScanLSTM(ScanLayer, FullyConnectedLayer):
    """
        LSTM implemented with the matrix being 4 times the dimensions so it can
        be sliced between the 4 gates.
    """
    def __init__(self, **kwargs):
        ScanLayer.__init__(self, kwargs.pop('spatial_input_dims', None))
        FullyConnectedLayer.__init__(self, **kwargs)

    # small attribute reminder
    def _get_activation(self):
        return None
    def _set_activation(self, value):
        if value is not None:
            raise AttributeError
    activation = property(_get_activation, _set_activation)

    def attribute_error(self, attr_name):
        if attr_name == 'activation':
            message = "trying to set an activation to " + \
                    self.__class__.__name__ + ". Does nothing as " + \
                    "LSTMs have gates hardcoded as sigmoid and tanh"
        else:
            message = 'default'
        super(ScanLSTM, self).attribute_error(attr_name, message)


    def batch_norm_addparams(self, param_dict):
        # 0.1 scaling as used in RNN BN paper
        param_dict.update({
            'x_gammas' : [(4*self.output_dims[0],), 'ones', 0.1],
            'h_gammas' : [(4*self.output_dims[0],), 'ones', 0.1],
            'c_gammas' : [self.output_dims, 'ones', 0.1],
            'c_betas' : [self.output_dims, 'zeros'],
        })


    @property
    def param_dict_initialization(self):
        # default 0.1 scaling on U, if higher more chances of exploding gradient
        dict_of_init = {
            'U' : [(self.input_dims[0], 4*self.output_dims[0]), 'orth', 0.1],
            'xh_betas' : [(4*self.output_dims[0],), 'zeros'],
            'h0' : [(self.output_dims[0],), 'zeros'],
            'c0' : [(self.output_dims[0],), 'zeros'],
        }
        if self.batch_norm:
            self.batch_norm_addparams(dict_of_init)
        return dict_of_init


    def initialize(self, dims):
        super(ScanLSTM, self).initialize(dims)
        ### Forget gate bias init to 1
        forget_bias = self.xh_betas.get_value()
        forget_bias[self.output_dims[0]:2*self.output_dims[0]] = 1.
        self.xh_betas.set_value(forget_bias)


    def op(self, h, U):
        if h.ndim > 2 :
            h = h.flatten(2)
        preact = T.dot(h, U)
        return preact


    def step(self, x_,
             h_, c_,
             U, xh_betas,
             c_gammas=None, c_betas=None,
             h_gammas=None, x_gammas=None):
        deterministic = self.deterministic
        if x_.ndim == 4 and h_.ndim == 2:
            x_ = x_.flatten(2)

        def _slice(_x, n, dim):
            if _x.ndim == 4:
                return _x[:, n*dim:(n+1)*dim, :, :]
            elif _x.ndim == 3:
                return _x[n*dim:(n+1)*dim, :, :]
            elif _x.ndim == 2:
                return _x[:,n*dim:(n+1)*dim]

        #from theano.tests.breakpoint import PdbBreakpoint
        #bp = PdbBreakpoint('test')
        #dummy_h_, U, x_ = bp(1, dummy_h_, U, x_)

        preact = self.op(h_, U)

        #import ipdb; ipdb.set_trace()
        if False and self.batch_norm :
            if self.batch_norm_on_x:
                x_normal = self.bn(x_, xh_betas, x_gammas, '_x', deterministic)
                h_normal = self.bn(preact, 0, h_gammas, '_h', deterministic)
            else:
                if x_gammas is None:
                    x_normal = x_
                else:
                    x_normal = self.bn(x_, 0, x_gammas, '_x', deterministic)
                h_normal = self.bn(preact, xh_betas, h_gammas, '_h', deterministic)
            preact = x_normal + h_normal
        else :
            xh_betas = xh_betas.dimshuffle(*('x',0) + ('x',) * (preact.ndim-2))
            preact = x_ + preact
            preact = preact + xh_betas

        i = T.nnet.sigmoid(_slice(preact, 0, self.output_dims[0]))
        f = T.nnet.sigmoid(_slice(preact, 1, self.output_dims[0]))
        o = T.nnet.sigmoid(_slice(preact, 2, self.output_dims[0]))
        g = T.tanh(_slice(preact, 3, self.output_dims[0]))

        delta_c_ = i * g

        c = f * c_ + delta_c_

        if self.batch_norm :
            c_normal = self.bn(c, c_betas, c_gammas, '_c', deterministic)
            h = o * T.tanh(c_normal)
        else :
            h = o * T.tanh(c)

        return [h, c], []


    def get_outputs_info(self, n):
        outputs_info = [T.repeat(self.h0[None,...], n, axis=0),
                        T.repeat(self.c0[None,...], n, axis=0)]
        return outputs_info


    def before_scan(self, x, axis=1, outputs_info=None):
        n_sample = x.shape[axis]
        sequences = [x]
        outputs_info = self.get_outputs_info(n_sample) if outputs_info is None \
                else outputs_info
        super(ScanLSTM, self).before_scan(sequences, outputs_info)


    def after_scan(self, scanout, updates):
        scanout = super(ScanLSTM, self).after_scan(scanout, updates)
        # only return h
        return scanout[0]



class ScanConvLSTM(ScanLSTM, ConvLayer):
    """
            ConvLSTM implementation where the time dot product is changed by a
        convolution operator half padded (do not change the dimensions)

        Some parameters could technically not be subclassed here again, however
        kewords are more clear and explicit to conv operation
        (such as num_filters) instead of output_dims
    """
    def __init__(self, filter_size, num_filters, **kwargs):
        ScanLayer.__init__(self, kwargs.pop('spatial_input_dims', None))
        ConvLayer.__init__(self, filter_size, num_filters, **kwargs)
        self.padding = 'half'
        self.strides = (1,1)


    def batch_norm_addparams(self, param_dict):
        # 0.1 scaling as used in RNN BN paper
        param_dict.update({
            'x_gammas' : [(4*self.num_filters,), 'ones', 0.1],
            'h_gammas' : [(4*self.num_filters,), 'ones', 0.1],
            'c_gammas' : [(self.num_filters,), 'ones', 0.1],
            'c_betas' : [(self.num_filters,), 'zeros'],
        })


    @property
    def param_dict_initialization(self):
        dict_of_init = {
            'U' : [(self.num_filters*4, self.num_filters)+self.filter_size,
                   'orth', 0.1],
            'xh_betas' : [(4*self.num_filters,), 'zeros'],
            'h0' : [(self.num_filters,) + self.feature_size, 'zeros'],
            'c0' : [(self.num_filters,) + self.feature_size, 'zeros'],
        }
        if self.batch_norm:
            self.batch_norm_addparams(dict_of_init)
        return dict_of_init


    def op(self, h, U):
        if self.filter_size == (1,1) :
            preact = T.dot(h.flatten(2), U.flatten(2).dimshuffle(1,0))[:,:,None,None]
        else :
            preact = T.nnet.conv2d(h, U, border_mode='half')
        return preact





    # FIX THAT if needed
"""
    def apply_zoneout(self, step):

            Act kind of like a decorator around the step function
            of scan which will perform the zoneout.

            ***ASSUMPTIONS :
                - mask_h is the first element of the sequences list
                - mask_c is the second element of the sequences list
                - h_ is how previous hidden state is named in step signature
                - c_ is how previous cell state is named in step signature

        def zonedout_step(*args, **kwargs):
            mask_h = args[0]
            mask_c = args[1]
            # arglist is unaware of the mask as it is not in the
            # step function signature (thats why the +2)
            arglist = inspect.getargspec(step).args
            h_ = args[arglist.index('h_')+2]
            c_ = args[arglist.index('c_')+2]
            zoneout_flag = args[arglist.index('zoneout_flag')+2]

            args = args[2:]
            h, c = step(*args, **kwargs)

            zoneout_mask_h = T.switch(
                zoneout_flag,
                mask_h,
                T.ones_like(mask_h) * self.zoneout/10.)

            zoneout_mask_c = T.switch(
                zoneout_flag,
                mask_c,
                T.ones_like(mask_c) * self.zoneout)

            zonedout_h = h_ * zoneout_mask_h + h * (T.ones_like(zoneout_mask_h) - zoneout_mask_h)
            zonedout_c = c_ * zoneout_mask_c + c * (T.ones_like(zoneout_mask_c) - zoneout_mask_c)
            return zonedout_h, zonedout_c
        return zonedout_step
"""



if __name__ == '__main__':
    from deepmonster.adlf.activations import LeakyRectifier
    from deepmonster.adlf.extras import Flatten
    from deepmonster.adlf.initializations import Initialization, Gaussian, Orthogonal
    from deepmonster.adlf.network import Feedforward
    from deepmonster.adlf.simple import FullyConnectedOnLastTime, FullyConnectedLayer
    from deepmonster.adlf.rnn import ConvLSTM
    time_size = 5
    image_size = (15,15)
    batch_size = 32
    valid_batch_size = 128
    channels = 3
    leak = 0.2

    lr = 1e-4
    lr = theano.shared(np.float32(lr), name='learning_rate')

    config = {
        'batch_norm' : True,
        'weight_norm' : False,
        'use_bias' : True,
        'activation' : LeakyRectifier(leak=leak),
        'initialization' : Initialization({'W' : Gaussian(std=0.05),
                                           'U' : Orthogonal(0.05)}),
    }

    layers = [
        ConvLSTM(3, 16, strides=(1,1), padding='half', num_channels=channels, image_size=image_size),
        FullyConnectedOnLastTime(output_dims=256),
        Flatten(),
        FullyConnectedLayer(output_dims=10)
    ]
    classifier = Feedforward(layers, 'classifier', **config)
    classifier.initialize()


    print "Init Model"
    #theano.config.compute_test_value = 'warn'
    ftensor5 = T.TensorType('float32', (False,)*5)
    x = ftensor5('features')
    #x.tag.test_value=np.random.random((time_size,batch_size,channels,32,32)).astype(np.float32)
    y = T.imatrix('targets')

    # train graph
    preds = classifier.fprop(x, deterministic=False)
    cost = T.nnet.categorical_crossentropy(preds, y.flatten()).mean()
    cost.name = 'train_cost'
    missclass = T.neq(T.argmax(preds, axis=1), y.flatten()).mean()
    missclass.name = 'train_missclass'

    # test graph
    v_preds = classifier.fprop(x, deterministic=True)
    v_missclass = T.neq(T.argmax(v_preds, axis=1), y.flatten()).mean()
    v_missclass.name = 'valid_missclass'

    xnp = np.random.random((time_size, batch_size, channels,)+image_size).astype(np.float32)
    ynp = np.random.randint(0, 256, size=(batch_size, 1))
    f = theano.function([x,y],[preds, missclass, v_missclass])
    out = f(xnp, ynp)
    import ipdb; ipdb.set_trace()
