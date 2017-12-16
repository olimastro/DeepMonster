from baselayers import EmptyLayer
from convolution import ConvLayer
from extras import InputInjectingLayer, SummationLayer
from upsampling import BilinearUpsampling
from network import Feedforward
from wrapped import WrapFprop

class ResNet(Feedforward):
    """
        Utility class to create resnet blocks.
        For more custom skip connected networks see class below.
    """
    def __init__(self, filter_size, num_filters, residual_at=2, upsample=False,
                 downsample=False, image_size=None, num_channels=None, **kwargs):
        assert (upsample and downsample) is False

        kwargs.setdefault('padding', 'half')
        if downsample == True:
            strides = (2,2)
        else:
            strides = (1,1)

        init = ConvLayer(filter_size, num_filters, strides=strides, image_size=image_size,
                         num_channels=num_channels, **kwargs)

        kwargs.setdefault('strides', (1,1))
        if upsample:
            layers = [BilinearUpsampling(2), init]
        else:
            layers = [init]
        layers = layers + [ConvLayer(filter_size, num_filters, **kwargs) \
                           for i in range(residual_at-1)]

        super(ResNet, self).__init__(layers, '', no_init=True, set_attr=False)


    # remember that his is propagated as is _fprop
    def set_attributes(self, i, layer, dict_of_hyperparam):
        layer.prefix = self.prefix + '_' + str(i)
        layer.set_attributes(dict_of_hyperparam)


    def fprop(self, x, **kwargs):
        y = super(ResNet, self).fprop(x, **kwargs)
        return x + y


# note to self: this is starting to look like the monster I have in mind,
# these classes could be expanded a little bit to be able to create huge pieces of
# networks instead of always linking multiple Feedforward graphs by hand.
class SkipConnectedHelper(object):
    """
        Small helper classes to ease the connection.

        You can either tag a layer with expose tag so it can
        be retrieved latter on the retrieve tag in the layer chain.
        OR expose_tag can be negative. When it is, it will ignore
        retrieve_tag and that this layer wants a connection from
        this relative position

        ex.:
        layers = [
            ConvLayer(3,16,num_channels=chan,image_size=imgs),
            ConvLayer(3,24),
            ConvLayer(3,16),
            SkipConnectedHelper(ConvLayer(3,32), -3, None, 'additive'), <--- 'a'
            ConvLayer(3,32),
            SkipConnectedHelper(None, -2, skip_type=MultiplicationLayer()) <--- 'b'
        ]
        SkipConnectedHelper 'a' will connect what goes OUT layerid (or position) -3
        relative to its own postion (3) with an additive connection to its INPUT.
        In other words, what goes OUT of layer0 is summed with what goes OUT of
        layer 2. SkipConnectedHelper 'b' will connect what goes OUT of layer 3 and
        what goes OUT of layer 4 (or IN layer 5) with a multiplication.
    """
    def __init__(self, layer=None, expose_tag=None, retrieve_tag=None, skip_type='additive'):
        assert skip_type == 'additive' or issubclass(skip_type.__class__, InputInjectingLayer)

        # TODO: the monster linking
        if retrieve_tag is not None: raise NotImplementedError("One day...")
        if expose_tag >= 0: raise NotImplementedError("One day...")
        # for now, lets implement the easy relative one

        self.skip_type = SummationLayer() if skip_type == 'additive' else skip_type
        self.layer = EmptyLayer() if layer is None else layer
        self.expose_tag = expose_tag
        self.retrieve_tag = retrieve_tag


class SkipConnectedNetwork(Feedforward):
    """
        Network object that will automatically apply required skip connection.
        It otherwise behaves as a Feedforward.

        skip_mapping: dict of format {layer i to apply: layer j to connect}
                      or None. If None, the user can provide the mapping
                      directly in the list of layers.

                      ex.: layers = [(Conv, 'a'), Conv, (Conv, 'a')]
                      will connect layer 0 to 2
        ***Therefore i > j all the time (TODO this check)

        skip_type: type of skip (additive (default)) or any other instance
        of InputInjectingLayer class
    """
    def __init__(self, layers, prefix, skip_mapping=None, **kwargs):
        super(SkipConnectedNetwork, self).__init__(layers, prefix, no_init=True, set_attr=False)

        # the user can
        if skip_mapping is None:
            self.skip_mapping = {}
            self.setup_mapping()
        else:
            self.skip_mapping = skip_mapping

        # Now we need to wrap all the layers that will get skip.
        # We use wrapping instead of simply inserting the inpuinjectinglayer
        # because it will preserve the original layers id
        self.wrap_up()

        # the helper class has to be gone
        assert all([not isinstance(x, SkipConnectedHelper) for x in self.layers])

        # only now can we initialize
        # TODO: write some kind of master method to wrap these calls
        self.set_attributes(kwargs)
        self.set_io_dims()
        self.initialize()
        self._has_been_init = True


    # ---- THESE METHODS ARE PROPAGATED WHEN CALLED ----
    def setup_mapping(self, i, layer):
        if isinstance(layer, SkipConnectedHelper):
            relative_i = layer.expose_tag
            self.skip_mapping.update({i: i + relative_i})


    def wrap_up(self, i, layer):
        if self.skip_mapping.has_key(i):
            self.layers[i] = WrapFprop(
                layer.layer, layer.skip_type)


    def clear_extra_inputs(self, i , layer):
        try:
            layer.clear_extra_inputs()
        except AttributeError:
            pass


    def _fprop(self, i, layer, **kwargs):
        #import ipdb; ipdb.set_trace()
        input_id = kwargs.pop('input_id', 0)
        if i < input_id:
            return
        # kwargs filtering
        for keyword in kwargs.keys():
            if keyword not in layer.accepted_kwargs_fprop:
                kwargs.pop(keyword)

        y = layer.fprop(self.activations_list[-1], **kwargs)

        # only if this layer's output has to be connected to
        # upstream layers do we need to do something different
        #import ipdb; ipdb.set_trace()
        if i in self.skip_mapping.values():
            for i_apply, i_connect in self.skip_mapping.iteritems():
                if i == i_connect:
                    self.layers[i_apply].add_extra_input(y)

        self.activations_list.append(y)
    # ------------------------------------------------- #


    def fprop(self, *args, **kwargs):
        rval = super(SkipConnectedNetwork, self).fprop(*args, **kwargs)
        # Better to clear connections so at next fprop call (if any) on
        # this network the whole chain gets built again
        self.clear_extra_inputs()
        return rval



if __name__ == "__main__":
    import theano
    import theano.tensor as T
    from utils import getnumpyf32
    from activations import Rectifier
    from extras import MultiplicationLayer

    imgs = (64,64)
    bs = 50
    chan = 3

    theano.config.compute_test_value = 'warn'
    x = T.ftensor4('x')
    npx = getnumpyf32((bs, chan,)+imgs)
    x.tag.test_value = npx

    config = {
        'batch_norm' : True,
        'activation' : Rectifier(),
        'padding': 'half',
    }

    #layers = [
    #    ConvLayer(3,16,num_channels=chan,image_size=imgs),
    #    ResNet(3, 16),
    #    ResNet(3, 16, residual_at=3)
    #]
    #resnet = Feedforward(layers, 'resnet', **config)
    layers = [
        ConvLayer(3,16,num_channels=chan,image_size=imgs),
        ConvLayer(3,24),
        ConvLayer(3,16),
        SkipConnectedHelper(ConvLayer(3,32), -3, None, 'additive'),
        ConvLayer(3,32),
        SkipConnectedHelper(None, -2, skip_type=MultiplicationLayer())
    ]
    resnet = SkipConnectedNetwork(layers, 'resnet', **config)

    y = resnet.fprop(x)
    f = theano.function([x],[y])
    print f(npx)[0].shape
    import ipdb; ipdb.set_trace()
