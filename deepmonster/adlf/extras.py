import theano
import theano.tensor as T

from baselayers import AbsLayer, WrappedLayer


class Reshape(AbsLayer):
    def __init__(self, shape, **kwargs):
        # use None in shape to use an unknown in advance input shape
        self.shape = shape
        super(Reshape, self).__init__(**kwargs)


    def set_io_dims(self, tup):
        self.input_dims = tup
        if len(self.shape) == 4:
            self.output_dims = self.shape[1:]
        if len(self.shape) == 5:
            self.output_dims = self.shape[2:]


    def apply(self, x):
        shape = []
        for i, shp in enumerate(self.shape):
            if shp is None:
                shape += [x.shape[i]]
            else :
                shape += [shp]
        return x.reshape(tuple(shape))



class Flatten(AbsLayer):
    # for now, only flat anything to bc
    def set_io_dims(self, tup):
        self.input_dims = tup
        od = 1
        for dim in tup:
            od *= dim
        self.output_dims = (od,)


    def apply(self, x):
        return T.flatten(x, outdim=2)



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
    from simple import FullyConnectedLayer

    x = T.ftensor4('x')
    y = T.ivector('y')
    lay = AddConditioning(FullyConnectedLayer(input_dims=25,output_dims=50),
                           nb_class=5)
    z = lay.fprop(x, y)
    f = theano.function([x,y],[z])
    npx = np.random.random((10,40,4,4)).astype(np.float32)
    npy = np.asarray([1,1,1,3,3,3,4,4,2,2]).astype(np.int32)
    out = f(npx,npy)
    import ipdb; ipdb.set_trace()
