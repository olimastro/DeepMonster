import copy
import theano
import theano.tensor as T

import utils
from activations import Activation
from graph import graph_traversal
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
        """
            Feedforward will call on each layer set_attributes with a dictionnary
            of all the values it has to set to all layers in the feedforwad block.

            By default, a layer has no attributes and will just pass this call
        """
        pass


    def initialize(self):
        """
            Initialize the values of the parameters according to their param_dict.
            In this dict, each parameter key map to a initilization method and
            an attribute self.key will be set.
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
            (class in network) will call on each layer.
        """
        return self.apply(*args, **kwargs)


    def param_dict_initialization(self):
        """
            Every layer should build a dict mapping
            name of parameters to their shapes of the format :

            {name : [shape, init_method, scaling_factor]}

            This is used at initialize() to setvalues to the parameters
        """
        self.param_dict = {}


    def apply(self, *args):
        raise NotImplementedError("Apply was called on layer {} {} and is not implemented".format(
            self, getattr(self, 'prefix', '')))


    @property
    def accepted_kwargs_fprop(self):
        """
            This function check every method fprop / apply of all child classes and return
            the set of keywords allowed in the calls.
        """
        mro = self.__class__.mro()
        kwargs = set()
        for cl in mro:
            if hasattr(cl, 'fprop'):
                kwargs.update(utils.find_func_kwargs(cl.fprop))
            if hasattr(cl, 'apply'):
                kwargs.update(utils.find_func_kwargs(cl.apply))
        return kwargs



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

        All kwargs should be None for reasons explaned in method set_attributes.
        Design choices exceptions: use_bias and attr_error_tolerance.
    """
    def __init__(self, attr_error_tolerance='warn', initialization=Initialization({}),
                 prefix=None, use_bias=True, batch_norm=None, activation=None,
                 weight_norm=None, train_g=None, **kwargs):
        super(Layer, self).__init__(**kwargs)

        self.attr_error_tolerance = attr_error_tolerance
        self.initialization = initialization

        self.prefix = prefix
        self.use_bias = use_bias
        self.batch_norm = batch_norm
        self.activation = activation

        self.weight_norm = weight_norm
        self.train_g = train_g


    def set_attributes(self, dict_of_hyperparam) :
        """
            The Layer class works in tandem with the Feedforward class. When a FF
            class is instanciated, it will call set_attributes on its list of
            layers. It will pass a set of kwargs to each layer in the list.
            The main feature here follows a simple rule: If a Layer was instanciated
            with an explicit keyword, it will preserve this keyword value. If
            not, this method will try to set it from the set of kwargs given
            to the FF instance.
        """
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
        if self.batch_norm and not self.weight_norm:
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
            except TypeError:
                import ipdb ; ipdb.set_trace()
                raise TypeError("Key: "+ self.prefix+key +" caused an error in initialization")
            param = theano.shared(param, name='%s_'%self.prefix+key)
            setattr(self, key, param)
            self.params += [param]

        if self.weight_norm:
            weight_norm(self, self.train_g)

        if self.batch_norm == 'mean_only':
            self.batch_norm = True
            self.bn_mean_only = True
        else :
            self.bn_mean_only = False


    def apply_bias(self, x):
        """
            Why a method even for this?? Because for example convolution
            can change this application if the bias is tied!
        """
        pattern = ['x'] * x.ndim
        if x.ndim in [3, 5]:
            # tbc or tbc01
            i = 2
        elif x.ndim in [2, 4]:
            # bc or bc01
            i = 1
        pattern[i] = 0
        pattern = tuple(pattern)

        return x + self.betas.dimshuffle(*pattern)


    def fprop(self, x, wn_init=False, deterministic=False, **kwargs):
        """
            fprop of this class is meant to deal with all the various inference / training phase
            or normalization scheme a layer could want
        """

        # deterministic is a 'fundamental' keyword and potentially can be used or not
        # used everywhere. The apply of a child method doesn't have to bother
        # all the time with it.
        kwargs.update({'deterministic':deterministic})
        try:
            preact = self.apply(x, **kwargs)
        except TypeError as e:
            if 'deterministic' in e.message:
                det = kwargs.pop('deterministic', False)
                preact = self.apply(x, **kwargs)
            else:
                raise e

        if self.batch_norm:
            preact = self.bn(preact, deterministic=deterministic)
        if wn_init:
            preact = self.init_wn(preact)

        if not self.batch_norm and self.use_bias:
            preact = self.apply_bias(preact)

        if self.activation is not None:
            return self.activation(preact)
        return preact


    def attribute_error(self, attr_name, message='default'):
        if message == 'default':
            message = "trying to set layer "+ self.__class__.__name__ + \
                    " with attribute " + attr_name
        if self.attr_error_tolerance is 'warn' :
            print "WARNING:", message
        elif self.attr_error_tolerance is 'raise' :
            raise AttributeError(message)


    # -------- Normalization related functions -------------- #
    ### batch norm ###
    def batch_norm_addparams(self):
            self.param_dict.update({
                'gammas' : [self.output_dims[0], 'ones']
            })


    # for this class, this is a useless wrapper to _bn_vars.
    # for RNN, both are not the same
    #@property
    #def bn_vars(self):
    #    return getattr(self, '_bn_vars')

    #@property
    #def _bn_vars(self):
    #    return getattr(self, '_bn_vars', [])

    #@_bn_vars.setter
    #def _bn_vars(self, value):
    #    self._bn_vars = value


    def add_bn_vars(self, var, name):
        var.name = name
        # check for key collision
        #for v in self._bn_vars:
        #    if name == v.name:
        #        raise RuntimeError("{} being reused in batch norm".format(name)+\
        #                           ", please make all the keys unique")
        setattr(var.tag, 'bn_statistic', name)
        #self._bn_vars.append(var)


    #def fetch_bn_vars(self, key):
    #    for v in self.bn_vars:
    #        if key in v.tag.bntag and 'mean' in v.tag.bntag:
    #            mean = v
    #        if key in v.tag.bntag and 'invstd' in v.tag.bntag:
    #            invstd = v
    #    if mean is None or invstd is None:
    #        raise RuntimeError("Could not fetch batch norm variables!")
    #    # cannot use premade types since we don't know in advance
    #    rmean = T.TensorType('float32', (False,) * mean.ndim)(
    #        self.prefix + '_' + mean.tag.bntag)
    #    rinvstd = T.TensorType('float32', (False,) * invstd.ndim)(
    #        self.prefix + '_' + mean.tag.bntag)

    #    # need to store them somewhere
    #    self.input_bn_vars = [rmean, rinvstd]
    #    return rmean, rinvstd


    def bn(self, x, betas=None, gammas=None, key='_', deterministic=False):
        """
            BN is the king of pain in the ass especially in the case of RNN
            (which is actually why this whole library was written for at first).
            We have two modes, training (deterministic=False) and testing
            (deterministic=True).

            The betas and gammas cannot always be fetched through self
            because of..... scan!!!!!
        """
        # make sure the format of the key is _something_
        if key != '_':
            if '_' != key[0]:
                key = '_' + key
            if '_' != key[-1]:
                key = key + '_'


        if deterministic:
            raise NotImplementedError("Dont do bn at det=True")
            mean, var = self.fetch_bn_vars(key)
        else:
            mean, var = (None, None,)

        if betas is None:
            betas = getattr(self, 'betas', 0.)
        if gammas is None:
            gammas = getattr(self, 'gammas', 1.)
        rval, mean, var = batch_norm(x, betas, gammas, mean, var)

        if not deterministic:
            self.add_bn_vars(mean, 'mean' + key + self.prefix)
            self.add_bn_vars(var, 'var' + key + self.prefix)

        return rval
    ###

    ### weight norm ###
    #FIXME: the dimshuffle on the mean and var depends on their dim.
    # Easy for 2&4D, but for a 5D or 3D tensor?
    def init_wn(self, x, init_stdv=0.1):
        raise NotImplementedError("You can use init_wn for now by doing batch +\
                                  norm on first layer")
        m = T.mean(x, self.wn_axes_to_sum)
        x -= m.dimshuffle(*self.wn_dimshuffle_args)
        inv_stdv = init_stdv/T.sqrt(T.mean(T.square(x), self.wn_axes_to_sum))
        x *= inv_stdv.dimshuffle(*self.wn_dimshuffle_args)
        self.wn_updates = [(self.betas, -m*inv_stdv), (self.g, self.g*inv_stdv)]

        return x
    # ------------------------------------------------------- #



class RecurrentLayer(AbsLayer):
    """
        A reccurent layer consists of two applications of somewhat independant
        layers: one that does an application like a feedforward, and the other
        that does the application through time.

        A rnn can also be used in mostly two fashion. It is either fed its own
        output for the next time step or it computes a whole sequence. In case
        1), we only need one theano scan which is outside what is actually just
        a normal FeedForwardNetwork. In case 2), every  single instance of an
        rnn needs to have its own theano scan.

        This class, with ScanLayer class, is intended to handle all these cases.
        NOTE: It should be possible to use a non scanlayer for the time application,
        in this  case if no step is implemented, this class will call the fprop
        of that layer.
    """
    def __init__(self, upwardlayer, scanlayer, mode='auto', time_collapse=True):
        assert mode in ['scan', 'out2in', 'auto']
        self.mode = mode
        self.upwardlayer = upwardlayer
        self.scanlayer = scanlayer
        self.time_collapse = time_collapse

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

    #@property
    #def bn_updates(self):
    #    updt = getattr(self.upwardlayer,'bn_updates',[])
    #    orderedupdt = getattr(self.scanlayer,'_updates',[])
    #    if isinstance(orderedupdt, list):
    #        updt += orderedupdt
    #    else:
    #        # it is an OrderedUpdate
    #        updt = updt + [item for item in orderedupdt.iteritems()]
    #    return updt

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


    def initialize(self):
        self.upwardlayer.initialize()
        self.scanlayer.initialize()


    def set_io_dims(self, tup):
        self.upwardlayer.set_io_dims(tup)
        self.scanlayer.set_io_dims(self.upwardlayer.output_dims)


    def fprop(self, x, outputs_info=None, **kwargs):
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

            # the outputinfo of the outside scan should contain the reccurent state
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
