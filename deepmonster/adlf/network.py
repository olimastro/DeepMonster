import inspect
import theano.tensor as T


class Feedforward(object):
    """
        Feedforward abstract class managing a series of Layer class.

        ***The logic of the attribution setting is always that if a hyperparameter
        such as batch_norm is set on a Layer, it will have priority
        over what is given to Feedforward constructor. If it is None and
        Feedforward receives something, it will set this value to the Layer.
        If a hyperparam is None in both Layer and Feedforward and Layer needs
        it to do something, it will obviously crash.

        WARNING: The __getattribute__ of this class is redefined so that the propagate
        method is applied as a decorator to each method of the class. It can have
        unexpected behaviors. If you want to protect a method (meaning it won't get
        propagated), put it in the list self.protected_method.
    """
    def __init__(self, layers, prefix, **kwargs):
        self.layers = layers
        self.prefix = prefix
        self.dict_of_hyperparam = kwargs
        self.fprop_passes = {}
        self.protected_method = [
            '__init__',
            'params',
            'propagate',
            'fprop',
            'input_dims',
            'output_dims',
            'get_outputs_info',
        ]

        set_attr = kwargs.pop('set_attr', True)
        if set_attr :
            self.set_attributes()

        # would be useful but try to not break all the scripts
        #if not kwargs.pop('no_init', False):
        #    self.initialize()


    def __getattribute__(self, name):
        """
            Applies self.propagate as a decorator to unprotected method
            in the class
        """
        def isprotected(name):
            return any([f == name for f in self.protected_method])

        if name == 'initialize':
            print "Initializing", self.prefix

        attr = object.__getattribute__(self, name)
        if hasattr(attr, '__call__') and not isprotected(name):
            def newfunc(*args, **kwargs):
                result = self.propagate(attr, *args, **kwargs)
                return result
            return newfunc
        return attr


    def dict_of_hyperparam_default(self) :
        """
            Every Feedforward instance should have its own
            default hyperparams attributes in order to work
        """
        pass


    # how to mix these two properties with the propagate decorator?
    @property
    def params(self):
        params = []
        for layer in self.layers :
            params += layer.params
        return params


    @property
    def outputs_info(self):
        outputs_info = []
        for layer in self.layers:
            if hasattr(layer, 'outputs_info'):
                outputs_info += layer.outputs_info
        return tuple(outputs_info)


    @property
    def output_dims(self):
        return self.layers[-1].output_dims

    @property
    def input_dims(self):
        return self.layers[-1].input_dims


    def propagate(self, func, *args, **kwargs):
        """
            Class decorator that takes a func to apply to every layer in the Feedforward chain.
        """
        for i, layer in enumerate(self.layers):
            func(i, layer, *args, **kwargs)


    # ---- THESE METHODS ARE PROPAGATED WHEN CALLED ----
    # exemple : foo = Feedforward(layers, 'foo', **fooconfig)
    #           foo.switch_for_inference()
    #           will propagate switch_for_inference to all layers
    def set_attributes(self, i, layer) :
        layer.prefix = self.prefix + str(i)
        layer.set_attributes(self.dict_of_hyperparam)


    def initialize(self, i, layer, **kwargs):
        if i == 0 :
            if not hasattr(layer, 'input_dims'):
                raise ValueError("The very first layer of this chain needs its input_dims!")
            layer.set_io_dims(layer.input_dims)
        else:
            layer.set_io_dims(self.layers[i-1].output_dims)
        layer.initialize(**kwargs)


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


    def _get_outputs_info(self, i, layer, *args, **kwargs):
        if hasattr(layer, 'get_outputs_info'):
            self._outputs_info += layer.get_outputs_info(*args, **kwargs)
    # ------------------------------------------------- #


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
            if not (isinstance(val, list) or isinstance(val, tuple)) or not isinstance(val[1], int):
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
        else:
            return self.activations_list[output_id]


    def get_outputs_info(self, *args, **kwargs):
        self._outputs_info = []
        self._get_outputs_info(*args, **kwargs)
        return self._outputs_info



if __name__ == '__main__':
    from simple import FullyConnectedLayer
    lay = [
        FullyConnectedLayer(input_dims=45, output_dims=50)
    ]
    feedfor = Feedforward(lay, 'ok', **{})
    import ipdb; ipdb.set_trace()
