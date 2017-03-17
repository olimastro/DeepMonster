import copy
import theano
import theano.tensor as T

import utils
from activations import Activation
from initializations import Initialization
from normalizations import weight_norm, batch_norm


class AbsLayer(object):
    """
        Semi-Abstract class which every layer should inherit from
    """
    def __init__(self, input_dims=None, output_dims=None):
        self.input_dims = utils.parse_tuple(input_dims)
        self.output_dims = utils.parse_tuple(output_dims)


    def set_attributes(self, attributes) :
        pass


    def initialize(self):
        """
            Initialize the values of the parameters according to their param_dict. In this dict,
            each parameter key map to a initilization method and an attribute self.key
            will be set
        """
        self.params = []


    def set_io_dims(self, tup):
        """
            This set the input output dims of the layer. By default input_dims = output_dims
            ***The next layer to fprop will use this layer's output_dims!

            NOTE: The activation might change the output_dims.
        """
        # every layer except the first one won't have this one set
        # except it was inforced by the user
        if not hasattr(self, 'input_dims') or self.input_dims[0] is None:
            self.input_dims = tup
        # by default the shapes don't change
        if not hasattr(self, 'output_dims') or self.output_dims[0] is None:
            self.output_dims = tup
        if hasattr(self, 'activation') and hasattr(self.activation, 'get_outdim'):
            new_dim = self.activation.get_outdim(self.output_dims)
            self.output_dims = new_dim


    def fprop(self, *args, **kwargs):
        """
            The propagation through the network is defined by every instance
            of fprop of Layer (see Layer subclass). This is what Feedforward
            (class below) will call on each layer.
        """
        raise NotImplementedError("An fprop on this layer should not have been called!")


    def param_dict_initialization(self):
        """
            Every layer should build a dict mapping
            name of parameters to their shapes of the format :

            {name : [shape, init_method, scaling_factor]}

            This is used at initialize() to setvalues to the parameters
        """
        self.param_dict = {}



class RandomLayer(AbsLayer):
    def __init__(self, seed=1234, **kwargs):
        self.seed = seed
        # you can use this class as a second inheritence  lets not try to init twice AbsLayer
        if not hasattr(self, "input_dims"):
            super(RandomLayer, self).__init__(**kwargs)
        self.rng_theano = theano.sandbox.rng_mrg.MRG_RandomStreams(seed)



class Layer(AbsLayer):
    """
        Layer class with initializable parameters and possible normalizations
    """
    def __init__(self, attr_error_tolerance='warn', initialization=Initialization({}),
                 prefix=None, use_bias=None, batch_norm=None, gamma_scale=None, activation=None,
                 weight_norm=None, train_g=None, **kwargs):
        super(Layer, self).__init__(**kwargs)

        self.attr_error_tolerance = attr_error_tolerance
        self.initialization = initialization

        self.prefix = prefix
        self.use_bias = use_bias
        self.batch_norm = batch_norm
        self.gamma_scale = gamma_scale
        self.activation = activation

        self.weight_norm = weight_norm
        self.train_g = train_g

        if self.batch_norm == 'mean_only':
            self.batch_norm = True
            self.bn_mean_only = True
        else :
            self.bn_mean_only = False
        self.record_batch_norm_mean = False


    def set_attributes(self, dict_of_hyperparam) :
        for attr_name, attr_value in dict_of_hyperparam.iteritems() :
            # if attr_name is set to a layer, it will keep that layer's attr_value
            try :
                attr = getattr(self, attr_name)
                if attr is None:
                    if isinstance(attr_value, Activation):
                        # make sure every layer has its own unique instance of the class
                        attr_value = copy.deepcopy(attr_value)
                    setattr(self, attr_name, attr_value)

                elif isinstance(attr, tuple):
                    # a (None,) wont trigger at the first if, but it doesn't count!
                    if attr[0] is None:
                        setattr(self, attr_name, utils.parse_tuple(attr_value, len(attr)))

                elif isinstance(attr, Initialization):
                    # this is a special case where the default is not None
                    if not len(self.initialization.vardict) > 0 :
                        setattr(self, attr_name, attr_value)

            except AttributeError :
                self.attribute_error(attr_name)


    def initialize(self) :
        self.params = []
        self.param_dict_initialization()
        if self.batch_norm:
            self.batch_norm_addparams()

        for key, value in self.param_dict.iteritems() :
            try :
                if len(value) < 3 or value[2] is None :
                    scaling = 1
                else :
                    scaling = value[2]

                if self.initialization.has_var(key):
                    param = self.initialization.get_init_tensor(key, value[0])
                else:
                    param = self.initialization.get_old_init_method(value[1], value[0], scaling)
                #param = initialization_method[value[1]](value[0]) * scaling
            except TypeError:
                import ipdb ; ipdb.set_trace()
                raise TypeError("Key: "+ self.prefix+key +" caused an error in initialization")
            param = theano.shared(param, name='%s_'%self.prefix+key)
            setattr(self, key, param)
            self.params += [param]

        #self.pop_mu = theano.shared(0., name='pop_mu')
        if self.weight_norm:
            weight_norm(self, self.train_g)


    def apply_bias(self, x):
        """
            Why a method even for this?? Because for example convolution
            can change this application if the bias is tied!
        """
        pattern = list(('x',) * x.ndim)
        if x.ndim in [3, 5]:
            # tbc or tbc01
            i = 2
        elif x.ndim in [2, 4]:
            # bc or bc01
            i = 1
        pattern[i] = 0
        pattern = tuple(pattern)

        return x + self.betas.dimshuffle(*pattern)


    def fprop(self, x, **kwargs):
        """
            fprop of this class is meant to deal with all the various inference / training phase
            or normalization scheme a layer could want
        """
        det = kwargs.get('deterministic', False)
        wn_init = kwargs.pop('wn_init', False)

        preact = self.apply(x, **kwargs)

        if self.batch_norm:
            preact = self.bn(preact, self.betas, self.gammas, deterministic=det)
        if wn_init:
            preact = self.init_wn(preact)

        if not self.batch_norm and self.use_bias:
            preact = self.apply_bias(preact)

        if self.activation is not None:
            return self.activation(preact)
        return preact


    def attribute_error(self, attr_name):
        message = "trying to set layer "+ self.__class__.__name__ + \
                " with attribute " + attr_name
        if self.attr_error_tolerance is 'warn' :
            print "WARNING:", message
        elif self.attr_error_tolerance is 'raise' :
            raise AttributeError(message)


    # -------- Normalization related functions -------------- #
    def batch_norm_addparams(self):
            self.param_dict.update({
                'gammas' : [self.output_dims[0], 'ones', self.gamma_scale]
            })


    def bn(self, x, betas, gammas, key='', deterministic=False):
        if deterministic:
            print "hoollallalaa"
            return (x-self.avg_batch_mean) / T.sqrt(1e-6 + self.avg_batch_var)
        rval, mean, var = batch_norm(x, betas, gammas, self.bn_mean_only)
        if not hasattr(self, 'avg_batch_mean'):
            self.avg_batch_mean = T.zeros_like(mean)
            self.avg_batch_var = T.ones_like(var)
            self.avg_batch_mean.name = self.prefix + key + "_avg_batch_mean"
            self.avg_batch_var.name = self.prefix + key + "_avg_batch_var"

            new_m = 0.9 * self.avg_batch_mean + 0.1 * mean
            new_v = 0.9 * self.avg_batch_var + 0.1 * var
            self.bn_updates = [(self.avg_batch_mean, new_m), (self.avg_batch_var, new_v)]
        return rval


    #FIXME: the dimshuffle on the mean and var depends on their dim. Easy for 2&4D, but for a 5D or 3D tensor?
    def init_wn(self, x, init_stdv=0.1):
        raise NotImplementedError("You can use wn for now by doing batch norm on first layer")
        m = T.mean(x, self.wn_axes_to_sum)
        x -= m.dimshuffle(*self.wn_dimshuffle_args)
        inv_stdv = init_stdv/T.sqrt(T.mean(T.square(x), self.wn_axes_to_sum))
        x *= inv_stdv.dimshuffle(*self.wn_dimshuffle_args)
        self.wn_updates = [(self.betas, -m*inv_stdv), (self.g, self.g*inv_stdv)]

        return x
    # ------------------------------------------------------- #



class RecurrentLayer(AbsLayer):
    """
        A reccurent layer consists of two applications of somewhat independant layers: one
        that does an application like a feedforward, and the other that does the application
        through time.

        A rnn can also be used in mostly two fashion. It is either fed its own output for the
        next time step or it computes a whole sequence. In case 1), we only need one theano scan
        which is outside what is actually just a normal FeedForwardNetwork. In case 2), every
        single instance of an rnn needs to have its own theano scan.

        This class, with ScanLayer class, is intended to handle all these cases.
        NOTE: It should be possible to use a non scanlayer for the time application, in this
        case if no step is implemented, this class will call the fprop of that layer.
    """
    def __init__(self, upwardlayer, scanlayer, mode='auto'):
        assert mode in ['scan', 'out2in', 'auto']
        self.mode = mode
        self.upwardlayer = upwardlayer
        self.scanlayer = scanlayer

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


    def get_outputs_info(self, *args):
        return self.scanlayer.get_outputs_info(*args)


    def set_attributes(self, attributes):
        self.upwardlayer.set_attributes(attributes)
        self.scanlayer.set_attributes(attributes)


    def initialize(self):
        self.upwardlayer.initialize()
        self.scanlayer.initialize()


    def set_io_dims(self, tup):
        self.upwardlayer.set_io_dims(tup)
        self.scanlayer.set_io_dims(self.upwardlayer.output_dims)


    def fprop(self, x, **kwargs):
        """
            This fprop should deal with various setups. if x.ndim == 5 it is pretty easy,
            every individual fprop of the rnn should handle this case easily since the fprop
            through time has its own time implementation.

            if x.ndim == 4 now it gets funky. Are we inside a for loop or a inside a theano scan?
            Since a for loop is easier, lets consider the scan case and the for loop user shall adapt.
            In this case kwargs should contain outputs_info which IN THE SAME ORDER should correspond
            to the reccurent state that the scanlayer.step is using.
        """
        # we don't want to give this to upwardlayer
        outputs_info = kwargs.pop('outputs_info', None)

        # logic here is that if x.ndim is 2 or 4, x is in bc or bc01
        # for 3 or 5 x is in tbc or tbc01. When t is here, you want to
        # scan on the whole thing.
        if self.mode == 'auto':
            mode = 'scan' if x.ndim in [3, 5] else 'out2in'
        else:
            mode = self.mode

        if x.ndim in [2, 4]:
            assert mode == 'out2in'
        # collapse batch and time together
        # TODO: modify this so the collapse is optional
        if x.ndim == 5:
            in_up  = x.reshape((x.shape[0]*x.shape[1],x.shape[2],x.shape[3],x.shape[4]))
        else:
            in_up = x
        h = self.upwardlayer.fprop(in_up, **kwargs)

        if mode == 'out2in':
            if not hasattr(self.scanlayer, 'step'):
                # hmm maybe this can work?
                return self.scanlayer.fprop(h)

            # the outputinfo of the outside scan should contain the reccurent state
            if outputs_info is None:
                raise RuntimeError("There should be an outputs_info in fprop of", self.prefix)
            outputs_info = list(outputs_info)

            # sketchy but not sure how to workaround
            self.scanlayer.deterministic = kwargs.get('deterministic', False)

            # this calls modify outputs info in the dict, but it should be fine
            self.scanlayer.before_scan(h, axis=0)
            args = tuple(self.scanlayer.scan_namespace['sequences'] + \
                         outputs_info + \
                         self.scanlayer.scan_namespace['non_sequences'])
            almosty = self.scanlayer.step(*args)

            # this is needed for the outside scan
            self.outputs_info = almosty

            y = self.scanlayer.after_scan(almosty)

        elif mode == 'scan':
            tup = (h.shape[-1],) if x.ndim == 3 else (h.shape[-3],h.shape[-2],h.shape[-1])
            h = h.reshape((x.shape[0], x.shape[1],)+tup)
            y = self.scanlayer.apply(h)

        return y



if __name__ == "__main__":
    from mlp import FullyConnectedLayer
    from convolution import ConvOnSequence
    fl = FullyConnectedLayer(input_size=20, output_size=30, prefix='fl',
                             weight_norm=True, train_g=True)
    fl.initialize()
    conv = ConvOnSequence(3, num_channels=10, num_filters=20,
                          mem_size_x=5, mem_size_y=5, prefix='conv',
                          weight_norm=True, train_g=True)
    conv.initialize()
    import ipdb; ipdb.set_trace()
