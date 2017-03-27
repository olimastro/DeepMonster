import inspect


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
        ]

        set_attr = kwargs.pop('set_attr', True)
        if set_attr :
            self.set_attributes()


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


    @property
    def params(self):
        params = []
        for layer in self.layers :
            params += layer.params
        return params


    def propagate(self, func, *args, **kwargs):
        """
            Class decorator that takes a func to apply to every layer in the Feedforward chain.
        """
        for i, layer in enumerate(self.layers):
            func(i, layer, *args, **kwargs)


    # THESE METHODS ARE PROPAGATED WHEN CALLED
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
        y = layer.fprop(self.activations_list[-1], **kwargs)
        self.activations_list.append(y)
    # -------------------------------------- #


    def fprop(self, x, output_id=-1, pass_name='', **kwargs):
        """
            Forward propagation passes through each layer

            inpud_id : use this index to start the fprop at that point in the feedforward block
            output_id : will return this index, can use 'all' for returning the whole list
            pass_name : if defined, will update the fprop_passes dict with {pass_name : activations_list}
        """

        self.activations_list = [x]
        self._fprop(**kwargs)

        if len(pass_name) > 0:
            self.fprop_passes.update({pass_name : self.activations_list})

        if output_id == 'all':
            return self.activations_list[1:]
        else:
            return self.activations_list[output_id]
