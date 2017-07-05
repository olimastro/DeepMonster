import theano
import theano.tensor as T

from baselayers import AbsLayer, WrappedLayer


class Reshape(AbsLayer):
    def __init__(self, shape, **kwargs):
        # mostly like lasagne
        # use -1 to collapse everything else on axis
        # use [i] on an element to put the original shape there
        self.shape = tuple(shape)
        if -1 in shape:
            assert shape[0] == -1 or shape[-1] == -1
            assert len(filter(lambda x: x == -1, shape)) == 1
        super(Reshape, self).__init__(**kwargs)


    def set_io_dims(self, tup):
        self.input_dims = tup
        if len(self.shape) == 4:
            self.output_dims = self.shape[1:]
        elif len(self.shape) == 5:
            self.output_dims = self.shape[2:]
        elif len(self.shape) in [2,3]:
            self.output_dims = (self.shape[-1],)
        for dim in self.output_dims:
            if dim == -1 or isinstance(dim, list):
                print "WARNING: Cannot safely infer output dims (channel, "+\
                        "height, width) of reshape layer {}, ".format(self.prefix)+\
                        "might crash if layer below depends on this."


    def apply(self, x):
        shape = []
        for i, shp in enumerate(self.shape):
            if isinstance(shp, list):
                shape += [x.shape[shp[0]]]
            else:
                shape += [shp]
        return x.reshape(tuple(shape))



class Flatten(AbsLayer):
    # for now, only flat height / width to channel
    def set_io_dims(self, tup):
        self.input_dims = tup
        od = 1
        for dim in tup:
            od *= dim
        self.output_dims = (od,)


    def apply(self, x):
        assert x.ndim == 4 or x.ndim == 5
        outdim = 2 if x.ndim == 4 else 3
        return T.flatten(x, ndim=outdim)



class SpatialMean(AbsLayer):
    def set_io_dims(self, tup):
        self.input_dims = tup
        self.output_dims = tup[0]+ (1,1)


    def apply(self, x):
        ndim = x.ndim - 1
        pattern = tuple(range(x.ndim-2)) + ('x','x')
        x = x.flatten(ndim=ndim)
        return T.mean(x, axis=ndim-1).dimshuffle(*pattern)



class AddConditioning(WrappedLayer):
    """
        Insert into the channel axis additionnal information.
        Typically used for example when wanting to condition
        the layer on the class information
    """
    def __init__(self, *args, **kwargs):
        self.nb_class = kwargs.pop('nb_class')
        super(AddConditioning, self).__init__(*args, **kwargs)


    @property
    def accepted_kwargs_fprop(self):
        kwargs = self.layer.accepted_kwargs_fprop
        kwargs.update('conditions_on')
        return kwargs


    # for now the only valid conditioning is the class
    # information being passed as integer, but it could be
    # extended.
    def fprop(self, x, conditions_on=None, **kwargs):
        i = -1 if x.ndim in [2,3] else -3
        shape = [x.shape[j] for j in range(x.ndim)]
        shape[i] = self.nb_class

        conditioning = T.zeros(tuple(shape), dtype=x.dtype)

        if x.ndim == 2:
            conditioning = T.inc_subtensor(conditioning[T.arange(x.shape[0]), conditions_on], 1.)
            x = T.set_subtensor(x[:,-self.nb_class:], conditioning)

        elif x.ndim == 3:
            conditioning = T.inc_subtensor(conditioning[:, T.arange(x.shape[0]), conditions_on], 1.)
            x = T.set_subtensor(x[:,:,-self.nb_class:], conditioning)

        elif x.ndim == 4:
            conditioning = T.inc_subtensor(conditioning[T.arange(x.shape[0]), conditions_on, :, :],
                                           1.)
            x = T.set_subtensor(x[:,-self.nb_class:,:,:], conditioning)

        elif x.ndim == 5:
            conditioning = T.inc_subtensor(conditioning[:, T.arange(x.shape[0]), conditions_on,:,:],
                                           1.)
            x = T.set_subtensor(x[:,:,-self.nb_class:,:,:], conditioning)

        return self.layer.fprop(x, **kwargs)



if __name__ == '__main__':
    import numpy as np
    #from convolution import ConvLayer5D as ConvLayer

    theano.config.compute_test_value = 'warn'
    npx = np.random.random((10,32,128,1,1)).astype(np.float32)
    ftensor5 = T.TensorType('float32', (False,)*5)
    x = T.ftensor5('x')
    x.tag.test_value = npx
    lay = Reshape((-1,128,1,1))
    y = lay.fprop(x)
    f = theano.function([x],[y])
    #out = f(npx,npy)
    out = f(npx)
    import ipdb; ipdb.set_trace()
