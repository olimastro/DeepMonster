import copy
from inspect import isclass
import theano.tensor as T
from deepmonster.utils import flatten
from deepmonster.nnet.baselayers import AbsLayer
from deepmonster.nnet.simple import BiasLayer


def propagate(func):
    """Network decorator to propagate a function call to all layers attribute of a class.
    """
    def propagate_func(*args, **kwargs):
        ff = args[0]
        for i, layer in enumerate(ff.layers):
            new_args = tuple([args[0], i, layer] + list(args[1:]))
            func(*new_args, **kwargs)
    return propagate_func


class Feedforward(object):
    """
        Feedforward abstract class managing a series of Layer class.

        ***The logic of the attribution setting is always that if a hyperparameter
        such as batch_norm is set on a Layer, it will have priority
        over what is given to Feedforward constructor. If it is None and
        Feedforward receives something, it will set this value to the Layer.
        If a hyperparam is None in both Layer and Feedforward and Layer needs
        it to do something, it will obviously crash.
    """
    def __init__(self, layers, prefix, **kwargs):
        self.layers = layers
        self.prefix = prefix
        self.fprop_passes = {}

        no_init = kwargs.pop('no_init', False)
        self.attr_error_tolerance = kwargs.pop('attr_error_tolerance', 'warn')

        if len(kwargs) > 1:
            self._set_attributes(kwargs)

        # would be useful but try to not break all the past scripts
        self._has_been_init = False
        if not no_init:
            #self.set_io_dims()
            self.initialize()
            self._has_been_init = True


    def __repr__(self):
        # printing an empty string would be quite boring
        if hasattr(self, 'prefix') and self.prefix != '':
            return self.prefix
        return super(Feedforward, self).__repr__()


    @property
    def params(self):
        return find_attributes(self.layers, 'params')

    # sugar syntax, but much needed sugar
    @property
    def parameters(self):
        return self.params

    @property
    def outputs_info(self):
        return tuple(find_attributes(self.layers, 'outputs_info'))


    @property
    def output_dims(self):
        return self.layers[-1].output_dims

    @property
    def input_dims(self):
        return self.layers[0].input_dims


    def _recurrent_warning(self, msg):
        # this is a poor way to do it but it works!
        if msg != getattr(self, 'last_msg', ''):
            print msg
            self.last_msg = msg


    # ---- THESE METHODS ARE PROPAGATED WHEN CALLED ----
    # exemple : foo = Feedforward(layers, 'foo', **fooconfig)
    #           foo.switch_for_inference()
    #           will propagate switch_for_inference to all layers
    @propagate
    def _set_attributes(self, i, layer, dict_of_hyperparam):
        if hasattr(layer, '_set_attributes'):
            layer._set_attributes(dict_of_hyperparam)
        self.set_attributes(layer, dict_of_hyperparam)

    @propagate
    def set_io_dims(self, i, layer, tup=None):
        if i == 0 :
            if not hasattr(layer, 'input_dims') and tup is None:
                raise ValueError("The very first layer of this chain needs its input_dims!")
            input_dims = getattr(layer, 'input_dims', (None,))
            if None in input_dims:
                dims = tup
            else:
                dims = input_dims
        else:
            dims = self.layers[i-1].output_dims
        layer.set_io_dims(dims)


    @propagate
    def initialize(self, i, layer, **kwargs):
        if self._has_been_init:
            msg = self.prefix + " already have been init, supressing this init call"
            self._recurrent_warning(msg)
            return

        layer.prefix = self.prefix + str(i)

        tup = kwargs.pop('tup', None)
        if i == 0 :
            if not hasattr(layer, 'input_dims') and tup is None:
                raise ValueError("The very first layer of this chain needs its input_dims!")
            input_dims = getattr(layer, 'input_dims', (None,))
            if None in input_dims:
                dims = tup
            else:
                dims = input_dims
        else:
            dims = self.layers[i-1].output_dims

        layer.initialize(dims, **kwargs)


    @propagate
    def _fprop(self, i, layer, **kwargs):
        input_id = kwargs.pop('input_id', 0)
        if i < input_id:
            return
        # kwargs filtering
        for keyword in kwargs.keys():
            if keyword not in layer.accepted_kwargs_fprop:
                kwargs.pop(keyword)

        if self.concatenation_tags.has_key(i):
            _input = T.concatenate([self.activations_list[-1]] +
                                   self.concatenation_tags[i][0],
                                   axis=self.concatenation_tags[i][1])
        else:
            _input = self.activations_list[-1]
        y = layer.fprop(_input, **kwargs)
        self.activations_list.append(y)


    @propagate
    def _get_outputs_info(self, i, layer, *args, **kwargs):
        if hasattr(layer, 'get_outputs_info'):
            self._outputs_info += layer.get_outputs_info(*args, **kwargs)
    # ------------------------------------------------- #

    def set_attributes(self, layer, dict_of_hyperparam):
        """
        """
        for attr_name, attr_value in dict_of_hyperparam.iteritems() :
            # if attr_name is set to a layer, it will keep that layer's attr_value
            try :
                attr = getattr(layer, attr_name)
            except AttributeError :
                self.attribute_error(layer, attr_name)
                continue

            if attr is None:
                if isinstance(attr_value, AbsLayer):
                    # make sure every layer has its own unique instance of the class
                    # deepcopy is very important or they might share unwanted stuff
                    # across layers (ex.: params)
                    attr_value = copy.deepcopy(attr_value)
                setattr(layer, attr_name, attr_value)

            elif isinstance(attr, tuple):
                # a (None,) wont trigger at the first if, but it doesn't count!
                if attr[0] is None:
                    setattr(layer, attr_name, utils.parse_tuple(attr_value, len(attr)))


    def attribute_error(self, layer, attr_name, message='default'):
        if message == 'default':
            message = "trying to set layer "+ layer.__class__.__name__ + \
                    " with attribute " + attr_name
        if self.attr_error_tolerance is 'warn' :
            print "WARNING:", message
        elif self.attr_error_tolerance is 'raise' :
            raise AttributeError(message)


    def fprop(self, x, output_id=-1, pass_name='', **kwargs):
        """
            Forward propagation passes through each layer

            inpud_id : use this index to start the fprop at that point in the feedforward block
            output_id : will return this index, can use 'all' for returning the whole list
            pass_name : if defined, will update the fprop_passes dict with {pass_name : activations_list}
        """
        # standarize the dict with {id of layer to inject input :
        # (list of tensors to concat, which axis)}
        concatenation_tags = kwargs.pop('concatenation_tags', {})
        assert all([isinstance(k, int) for k in concatenation_tags.keys()])
        for key, val in concatenation_tags.iteritems():
            if not isinstance(val, (list, tuple)) or len(val) == 1 or not isinstance(val[1], int):
                val = [val, None]
            else:
                assert len(val) == 2
            if not isinstance(val[0], list):
                val = [[val[0]], val[1]]
            assert len(set([v.ndim for v in val[0]])) == 1, "A list of tensors " +\
                    "to concat was given but not all have same dim"
            if val[1] is None:
                # default on the channel axis
                ndim = val[0][0].ndim
                val[1] = ndim - 3 if ndim in [4, 5] else ndim - 1
            concatenation_tags[key] = tuple(val)
        self.concatenation_tags = concatenation_tags

        self.activations_list = [x]
        self._fprop(**kwargs)

        if len(pass_name) > 0:
            self.fprop_passes.update({pass_name : self.activations_list})

        del self.concatenation_tags
        if output_id == 'all':
            return self.activations_list[1:]
        elif isinstance(output_id, list):
            return [self.activations_list[i] for i in output_id]
        else:
            return self.activations_list[output_id]


    def get_outputs_info(self, *args, **kwargs):
        self._outputs_info = []
        self._get_outputs_info(*args, **kwargs)
        return self._outputs_info



class StandardBlock(Feedforward):
    """Standard blocks of Layers module. Every piece of the Layer class could technically
    be used all seperatly, but this encapsulate the regular usage of layer i.e.:
        y = activation(normalization(apply(W,x) + b))

        - y: output
        - x: input
        - W: apply parameters
        - b: bias parameters
        - apply: some method coupling x and W together (ex.: FullyConnectedLayer: dot)
        - normalization: normalization class instnace normalizing the output of apply
            (ex.: BatchNorm)
        - activation: activation class instance applying a function before the output
            (ex.: Rectifier or ReLU)

    The block interface is designed to work with Feedforward in order to initialize multiple layers
    in a somewhat lazy way. For this it provides set_attributes method where it allows keywords not
    given at __init__ time to be given to Feedforward so it can propagate them all and set them on
    its list of layers. This comes with the cost that we do not know at __init__ time how to construct
    the block. It is therefore done in construct_block method that should ideally be called after
    __init__ and set_attributes.
    """
    apply_layer_type = NotImplemented

    # special kwargs shared with its apply layer
    shared_with_apply_kwargs = ['initialization', 'param_norm']

    def __init__(self, *args, **kwargs):
        # filter kwargs, after this it should contain only apply layer kwargs
        # and return the one for this class
        kwargs = self.parse_kwargs(**kwargs)
        self.set_apply_layer(args, kwargs)


    def parse_kwargs(self, bias=True, apply_layer=None, activation_norm=None,
                     activation=None, initialization=None, param_norm=None,
                     attr_error_tolerance='warn', apply_fetch=None, **kwargs):
        """Because def __init__(self, *args, a_kwarg=a_default_val, **kwargs) is a python
        syntax error, we cannot write the __init__ of StandardBlock this way. The constructor
        of StandardBlock pipes args and kwargs for the constructor of its apply Layer. Kwargs
        dedicated for itself are defined here and unpack while passing **kwargs to this method
        at __init__ time.
        """
        for k, v in locals().iteritems():
            if k == 'kwargs':
                continue
            setattr(self, k ,v)
        kwargs.update({x: getattr(self, x) for x in self.shared_with_apply_kwargs})
        return kwargs


    @property
    def layers(self):
        # block follow the strict layer order: apply, bias, act_norm, act
        layers = filter(
            lambda x: x is not None,
            [getattr(self, x, None) for x in ['apply_layer', 'bias_layer', 'activation_norm_layer', 'activation']])
        return layers


    def set_apply_layer(self, args, kwargs):
        """Set the apply layer
        """
        if self.apply_layer is None and self.apply_layer_type is NotImplemented:
            raise NotImplementedError("No apply layer given to construct this block")
        if self.apply_layer is not None and self.apply_layer_type is not NotImplemented:
            raise RuntimeError("Ambiguity while trying to construct a standard block")

        if isinstance(self.apply_layer, AbsLayer):
            return
        elif isinstance(self.apply_fetch, str):
            assert isinstance(self.apply_layer_type, dict), \
                    "Cannot fetch apply layer by string if its apply_layer_type is not implemented as a dict"
            ApplyLayer = self.apply_layer_type[self.apply_fetch]
        elif isclass(self.apply_layer):
            ApplyLayer = self.apply_layer
        elif self.apply_layer is None:
            if isinstance(self.apply_layer_type, dict):
                ApplyLayer = self.apply_layer_type['default']
            else:
                ApplyLayer = self.apply_layer_type
        else:
            raise ValueError("Does not recognize apply layer")
        self.apply_layer = ApplyLayer(*args, **kwargs)


    #def set_attributes(self, layer, dict_of_hyperparam):
    #    import ipdb; ipdb.set_trace()
    #    # since we are catching set_attributes, self and layer are this object
    #    # first set attributes on the whole block
    #    super(StandardBlock, self).set_attributes(layer, dict_of_hyperparam)
    #    # second propagate it down its own layers list
    #    super(StandardBlock, self)._set_attributes(dict_of_hyperparam)


    def get_layer(self, layerkey):
        layer_opt = getattr(self, layerkey)
        if layer_opt is None or layer_opt is False:
            return None
        elif isinstance(layer_opt, AbsLayer):
            return layer_opt
        elif layerkey == 'bias' and layer_opt is True:
            bias_kwargs = {x: getattr(self, x, None) for x in ['initialization', 'param_norm']}
            return BiasLayer(**bias_kwargs)
        elif isclass(layer_opt):
            raise ValueError(
                "A class {} was given for layer creation in block, needs an instance".format(layer_opt))
        raise ValueError("Does not recognize {}".format(layerkey))


    def initialize(self, *args, **kwargs):
        """Complete the block's initialization. ApplyLayer should already exists.

        Since some layers have parameters, Initialization and ParamNorm affects all ParametrizedLayer
        in the block.
        """
        # apply layer was already set
        # set bias layer
        if self.activation_norm is not None:
            self.bias = False
        self.bias_layer = self.get_layer('bias')
        # set activation norm layer
        self.activation_norm_layer = self.get_layer('activation_norm')
        # set activation layer
        self.activation_layer = self.get_layer('activation')
        #self.activation_layer = self.activation \
        #        if self.activation is not None else None

        # we can now safely propagate initialize call on this block of layers
        # this flag is for the inner layers init
        self._has_been_init = False
        tup = args[0]
        kwargs.update({'tup': tup})
        super(StandardBlock, self).initialize(**kwargs)



def find_attributes(L, a):
    # return a FLAT list of all attributes found
    if isinstance(L, set):
        L = list(L)
    elif not isinstance(L, (list, tuple)):
        L = [L]
    attributes = []
    for l in L:
        attributes += flatten(getattr(l, a, []))
    return attributes


def find_nets(localz):
    # this is for the lazy :)
    # give locals() as argument to the script defining the networks
    return [item for key, item in localz.iteritems() \
            if isinstance(item, Feedforward) and key != 'Feedforward']


if __name__ == '__main__':
    from simple import FullyConnectedLayer
    lay = [
        FullyConnectedLayer(input_dims=45, output_dims=50)
    ]
    feedfor = Feedforward(lay, 'ok', **{})
    import ipdb; ipdb.set_trace()
