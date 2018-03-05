import copy
import theano
import theano.tensor as T

import utils
from initializations import Initialization


class AbsLayer(object):
    """Semi-Abstract class which every layer should inherit from
    """

    def __init__(self, input_dims=None, output_dims=None, prefix=''):
        self.input_dims = utils.parse_tuple(input_dims)
        self.output_dims = utils.parse_tuple(output_dims)
        self.prefix = prefix


    def initialize(self, tup):
        """We usually do not know the dims when calling __init__ as output_dims
        is a function of input_dims.
        Ex.: The next layer to fprop will use this layer's output_dims!
        """
        self.set_io_dims(tup)


    def set_io_dims(self, tup):
        """By default input_dims = output_dims
        """
        # every layer except the first one won't have this one set
        # except it was inforced by the user
        if not hasattr(self, 'input_dims') or self.input_dims[0] is None:
            self.input_dims = tup
        # by default the shapes don't change
        if not hasattr(self, 'output_dims') or self.output_dims[0] is None:
            self.output_dims = tup


    def fprop(self, *args, **kwargs):
        """This is what Feedforward (class in network) will call on each layer. It can also be used
        individually. A layer has to implement fprop or apply (fprop is taken first) and it will
        call apply if it is not implemented in the chlid class.

        This is a flexibility design which allows to create layer classes that contains sub layers
        and output a mix of all their individual applies and / or fprop.
        """
        return self.apply(*args, **kwargs)


    def apply(self, *args):
        """ Apply method called by fprop"""
        # if we are here, it is an error
        raise NotImplementedError("Apply was called on layer {} {} and is not implemented".format(
            self, getattr(self, 'prefix', '')))


    @property
    def accepted_kwargs_fprop(self):
        """This function check every method fprop / apply of all child classes and return
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


class ParametrizedLayer(AbsLayer):
    """Layer class with parameters as given in its param_dict_initialization property.
    """

    def __init__(self, initialization=None, param_norm=None, **kwargs):
        self.params = []
        self.initialization = initialization
        self.param_norm = param_norm
        super(ParametrizedLayer, self).__init__(**kwargs)


    @property
    def param_dict_initialization(self):
        """Every layer should build a default dict mapping name of parameters to their shapes of
        the format:
            {name : [shape, init_method, scaling_factor]}

        This is used at initialize() to setvalues to the parameters as default behavior. It can
        be overridden by passing an Initialization object to the layer.
        """
        return NotImplemented


    def initialize(self, tup) :
        """Initialize parameters
        """
        # but first set the shape
        super(ParametrizedLayer, self).initialize(tup)

        params = getattr(self, 'params', [])
        if len(params) > 0:
            print "WARNING: initialize has been called on {} ".format(self.prefix) + \
                    "and it already contains some params. This call will overwrite them."

        for key, value in self.param_dict_initialization.iteritems() :
            try :
                if len(value) < 3 or value[2] is None :
                    scaling = 1
                else :
                    scaling = value[2]

                if self.initialization is None or not self.initialization.has_var(key):
                    param = Initialization.get_old_init_method(value[1], value[0], scaling)
                else:
                    param = self.initialization.get_init_tensor(key, value[0])
            except TypeError as e:
                import ipdb ; ipdb.set_trace()
                raise TypeError("Key: "+ self.prefix+key +" caused an error in initialization")

            #FIXME for new backends!
            param = theano.shared(param, name='%s_'%self.prefix+key)
            setattr(self, key, param)
            params += [param]

        # for now it doesn't need to be implemented as a class since there is only this one
        if self.param_norm == 'weight_norm':
            weight_norm(self, self.train_g)

        self.params = params


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


def weight_norm(layer, train_g=None):
    """
        Applies weight norm on a layer
        train_g None := no g at all
        train_g False := g gets a value but is not propagated on as a param
        train_g True := same as False but it is updated
    """
    assert train_g in [None, False, True]
    init_g = Constant(1.)

    try:
        weight_tag = 'W' if hasattr(layer, 'W') else 'U'
    except AttributeError:
        raise AttributeError("Trying to call weight norm on {} ".format(layer)+\
                             "without layer.W or layer.U defined")
    weights = getattr(layer, weight_tag)

    Wndim = weights.get_value().ndim
    if Wndim == 4:
        W_axes_to_sum = (1,2,3)
        W_dimshuffle_args = (0,'x','x','x')
    elif Wndim == 5:
        W_axes_to_sum = (1,2,3,4)
        W_dimshuffle_args = (0,'x','x','x','x')
    # a bit sketch but serves our purpose for the LSTM weights
    #elif weight_tag == 'U' and Wndim == 2:
    #    W_axes_to_sum = 1
    #    W_dimshuffle_args = (0,'x')
    elif Wndim == 3 :
        raise NotImplementedError("What is a weight with 3 dimensions?")
    else :
        W_axes_to_sum = 0
        W_dimshuffle_args = ('x',0)

    if train_g is not None:
        g = init_g(layer.output_dims)
        g = theano.shared(g, name=layer.prefix+'_g')
        if train_g :
            layer.params += [g]

        new_weights = weights * (
             g / T.sqrt(1e-6 + T.sum(T.square(weights),
                                     axis=W_axes_to_sum))).dimshuffle(*W_dimshuffle_args)
        layer.g = g
    else:
        new_weights = weights / \
                T.sqrt(1e-6 + T.sum(T.square(weights),
                                    axis=W_axes_to_sum,keepdims=True))

    setattr(layer, weight_tag, new_weights)



