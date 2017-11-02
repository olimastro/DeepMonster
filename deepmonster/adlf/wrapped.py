import copy
import theano
import theano.tensor as T
from convolution import ConvLayer, DeConvLayer
from simple import FullyConnectedLayer
from utils import collapse_time_on_batch, expand_time_from_batch, find_func_kwargs


class WrappedLayer(object):
    """
        This class is used to intercept a normal application of a layer
        to do something else. This class owns almost nothing and returns
        attributes and methods of its layer.

        Note: One limitation is that a normal usecase will be done by writing
        an fprop method. Its own (not the layer's)
        accepted_kwargs_fprop will have to be case specific and written by hand.
    """
    def __init__(self, layer):
        self.setattr('layer', layer)


    def __getattribute__(self, name):
        """
            Any called attribute or method will return the one of the wrapped layer
            if it is not defined in this class.
        """
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return self.layer.__getattribute__(name)


    def __setattr__(self, name, value):
        """
            Any called attribute or method will return the one of the wrapped layer
            if it is not defined in this class.
        """
        setattr(self.layer, name, value)


    def setattr(self, name, value):
        """
            The way to set an attr directly on the wrapped class
        """
        return object.__setattr__(self, name, value)



class WrapTimetoBatch(WrappedLayer):
    """
        Class to wrap a tensor with a time axis to a non-time processing layer
        such as a CNN using the batch axis. The amount of time concatenated on the
        batch axis is important when using batch norm.
    """
    def __init__(self, layer, sliced_time=1, time_size=None):
        super(WrapTimetoBatch, self).__init__(layer)
        self.setattr('sliced_time', sliced_time)
        assert time_size is None or isinstance(time_size, int)
        if time_size is not None:
            assert time_size % sliced_time == 0
        self.setattr('time_size', time_size)


    def fprop(self, x, **kwargs):
        if x.ndim in (2,4):
            # does nothing
            return self.layer.fprop(x, **kwargs)
        if self.time_size is None:
            # do a full time collapse
            y, shp = collapse_time_on_batch(x)
            y = self.layer.fprop(y, **kwargs)
            return expand_time_from_batch(y, shp)
        else:
            rval = []
            for i in range(0, self.time_size, self.sliced_time):
                y, shp = collapse_time_on_batch(x[i:i+self.sliced_time])
                y = self.layer.fprop(y, **kwargs)
                rval.append(expand_time_from_batch(y, shp))

            return T.concatenate(rval, axis=0)


# sugar definitions
# some sugar config...
class predefinedWrappedConfig(object):
    """
        Config class. This is general and fancy enough to be moved eventually in a
        config script if I need more config across all files. For
        now only wrapped uses it.
    """
    __new_count = 0
    def __init__(self):
        object.__setattr__(self, '_config', copy.copy(self.__config))

    def __new__(cls, *args, **kwargs):
        cls.__new_count += 1
        if cls.__new_count > 1:
            raise RuntimeError("Cannot instantiate config more than once")
        else:
            return object.__new__(cls, *args, **kwargs)

    # default
    __config = {
        'time_size': None,
        'sliced_time': 1,
    }

    def __iter__(self):
        def config_iterator():
            for key in self._config.keys():
                yield key
        return config_iterator()

    @property
    def on_attr_change(self):
        return getattr(self, '_on_attr_change', 'warn')
    @on_attr_change.setter
    def on_attr_change(self, name):
        assert name in ['ignore','warn','raise','frozen']
        object.__setattr__(self, '_on_attr_change', name)

    def avail(self):
        return [(k,v) for k,v in self._config.iteritems()]

    def set(self, name, value):
        if self.on_attr_change == 'frozen':
            raise AttributeError('Config was set to freeze its values and {}'.format(name)+\
                                 ' is being changed.')
        if name not in self._config.keys():
            if self.on_attr_change == 'raise':
                raise AttributeError('Trying to set unrecognized config option')
            elif self.on_attr_change == 'warn':
                print 'Trying to set unrecognized config option, does nothing'
        else:
            if not self.on_attr_change == 'ignore':
                print "Changing config {} for {}".format(name, value)
            self._config[name] = value

    def __setattr__(self, name, value):
        # moar sugar!!!
        # this also freezes any setting on this class, which is wanted
        if name in self._config.keys():
            self.set(name, value)
        else:
            raise AttributeError('Trying to set unrecognized config option')

    def __getattr__(self, name):
        # note to self: remember that this is called when everything fails
        # compared to __getattribute__ which is called unconditionnally
        try:
            return self._config[name]
        except KeyError:
            raise AttributeError("No config named {} found".format(name))
config = predefinedWrappedConfig()


# ...for some sugary syntax!
class PredefinedWrappedLayer(object):
    """
        This class is to be used with child classes to provide
        easy to write classes for the user when using a wrapper.
        Ex.: You can just call ConvLayer5D to emulate the convlayer
        with its right init without having to write
        Wrapper(Conv(convargs, convkwargs), wrapperkwargs).
    """
    def __new__(cls, *args, **kwargs):
        Wrapper, Layer = cls.get_wrapped()
        keys_wrapper = find_func_kwargs(Wrapper.__init__)
        kwargs_wrapper = {key:kwargs.pop(key) for key in kwargs.keys() if key in keys_wrapper}
        cls.source_config(kwargs_wrapper)
        return Wrapper(Layer(*args, **kwargs), **kwargs_wrapper)

    @classmethod
    def source_config(cls, kwargs):
        for conf in config:
            if conf not in kwargs.keys():
                kwargs.update({conf: getattr(config, conf)})


class ConvLayer5D(PredefinedWrappedLayer):
    @classmethod
    def get_wrapped(cls):
        return WrapTimetoBatch, ConvLayer

class DeConvLayer5D(PredefinedWrappedLayer):
    @classmethod
    def get_wrapped(cls):
        return WrapTimetoBatch, DeConvLayer

class FullyConnectedLayer3D(PredefinedWrappedLayer):
    @classmethod
    def get_wrapped(cls):
        return WrapTimetoBatch, FullyConnectedLayer



if __name__ == "__main__":
    config.set('time_size', 4)
    config.sliced_time = 2
    foo = FullyConnectedLayer3D(input_dims=64, output_dims=128,
                                prefix='fl', batch_norm=True, use_bias=False)
    foo.initialize()
    from utils import getnumpyf32
    npx = getnumpyf32((4,3,64))
    x = T.ftensor3('x')
    y = foo.fprop(x)
    func = theano.function([x],[y])
    print func(npx)[0].shape
    import ipdb; ipdb.set_trace()
