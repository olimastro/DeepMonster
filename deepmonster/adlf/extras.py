import theano
import theano.tensor as T

from baselayers import AbsLayer


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


    def fprop(self, x):
        shape = []
        for i, shp in enumerate(self.shape):
            if shp is None:
                shape += [x.shape[i]]
            else :
                shape += [shp]
        return x.reshape(tuple(shape))



class SpatialMean(AbsLayer):
    def set_io_dims(self, tup):
        self.input_dims = tup
        self.output_dims = tup[0]+ (1,1)


    def fprop(self, x):
        ndim = x.ndim - 1
        pattern = tuple(range(x.ndim-2)) + ('x','x')
        x = x.flatten(ndim=ndim)
        return T.mean(x, axis=ndim-1).dimshuffle(*pattern)
