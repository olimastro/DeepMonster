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

        All kwargs should be None for reasons explained in method set_attributes.
        Design choices exceptions: use_bias and attr_error_tolerance.
    """
    def __init__(self, attr_error_tolerance='warn', initialization=Initialization({}),
                 prefix=None, use_bias=True, batch_norm=None, activation=None,
                 conditional_batch_norm=None, weight_norm=None, train_g=None, **kwargs):
        super(Layer, self).__init__(**kwargs)

        self.attr_error_tolerance = attr_error_tolerance
        self.initialization = initialization

        self.prefix = prefix
        self.use_bias = use_bias
        self.batch_norm = batch_norm
        self.conditional_batch_norm = conditional_batch_norm
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
            except TypeError as e:
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


    def delete_params(self, param_name, on_error='raise'):
        # param is the param ATTRIBUTE name such as self.W, not its
        # name as in the theano shared var name
        # make sure to remove everywhere it could appear
        try:
            param = getattr(self, param_name)
            self.params.remove(param)
            self.param_dict.pop(param_name)
            delattr(self, param_name)
            del param
        except Exception as e:
            if on_error == 'raise':
                raise e
            elif on_error == 'warn':
                print "WARNING: Error while deleting parameter {} on {}".format(
                    param_name, self.prefix) + ", ignoring."


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
            fprop of this class is meant to deal with all the various inference
            / training phase or normalization scheme a layer could want
        """

        # deterministic is a 'fundamental' keyword and potentially can be used or not
        # used everywhere. The apply of a child method doesn't have to bother
        # all the time with it.
        kwargs.update({'deterministic':deterministic})
        betas = kwargs.pop('betas', None)
        gammas = kwargs.pop('gammas', None)
        try:
            preact = self.apply(x, **kwargs)
        except TypeError as e:
            if 'deterministic' in e.message:
                det = kwargs.pop('deterministic', False)
                preact = self.apply(x, **kwargs)
            else:
                raise e

        if self.batch_norm or self.conditional_batch_norm:
            if self.conditional_batch_norm:
                assert betas is not None and gammas is not None
            preact = self.bn(preact, betas, gammas, deterministic=deterministic)
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


    def tag_bn_vars(self, var, name):
        # tag the graph so popstats can use it
        var.name = name
        setattr(var.tag, 'bn_statistic', name)


    def bn(self, x, betas=None, gammas=None, key='_', deterministic=False, axis='auto'):
        """
            BN is the king of pain in the ass especially in the case of RNN
            (which is actually why this whole library was written for at first).

            BN is to be used with get_inference_graph in popstats at inference phase.
            It will compute the batch statistics from a dataset and replace in the
            theano graph the tagged bnstat with the computed values.

            All the deterministic logic is therefore deprecated here.
        """
        # make sure the format of the key is _something_
        if key != '_':
            if '_' != key[0]:
                key = '_' + key
            if '_' != key[-1]:
                key = key + '_'

        #if deterministic:
        #    print "WARNING: deterministic=True is deprecated in Layer.bn and has"+\
        #            " no effect"

        mean, std = (None, None,)

        if betas is None:
            betas = getattr(self, 'betas', 0.)
        if gammas is None:
            gammas = getattr(self, 'gammas', 1.)
        rval, mean, std = batch_norm(x, betas, gammas, mean, std,
                                     cbn=self.conditional_batch_norm,
                                     mean_only=self.bn_mean_only,
                                     axis=axis)

        # do not tag on cbn
        if not deterministic and not self.conditional_batch_norm:
            self.tag_bn_vars(mean, 'mean' + key + self.prefix)
            self.tag_bn_vars(std, 'std' + key + self.prefix)

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
