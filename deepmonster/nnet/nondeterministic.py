import theano
import theano.tensor as T

from baselayers import RandomLayer
from deepmonster import config
floatX = config.floatX

class NonDeterministicLayer(RandomLayer):
    """
        Any layer subclassing this class should have limited applications
        such as dropout where it changes the theano graph only for
        deterministic == True / False
    """
    def fprop(self, x, deterministic=False, **kwargs):
        if deterministic:
            return x
        else:
            return self.apply(x, **kwargs)



class Dropout(NonDeterministicLayer):
    def __init__(self, p, mode='normal', **kwargs):
        super(Dropout, self).__init__(**kwargs)
        assert mode in ['normal', 'spatial']
        self.p = 1.- p
        self.mode = mode


    def apply(self, x):
        if self.mode is 'normal' or x.ndim == 2:
            shape = x.shape
            dropout_mask = self.rng_theano.binomial(x.shape, p=self.p, dtype=floatX) / self.p
        elif x.ndim == 4 or x.ndim == 5:
            # spatial dropout, meaning drop a whole feature map
            shape = (x.shape[x.ndim-3],)
            pattern = ('x',) * (x.ndim-3) + (0,) + ('x','x',)

            dropout_feature_mask = self.rng_theano.binomial(shape, p=self.p, dtype=floatX) / self.p
            dropout_mask = T.ones_like(x) * dropout_feature_mask.dimshuffle(*pattern) / self.p
        return x



class AdditiveGaussianNoise(NonDeterministicLayer):
    def __init__(self, std=1., **kwargs):
        super(AdditiveGaussianNoise, self).__init__(**kwargs)
        self.std = std


    def apply(self, x):
        noise = self.rng_theano.normal(size=x.shape) * self.std
        return x + noise



class MultiplicativeGaussianNoise(NonDeterministicLayer):
    def __init__(self, std=1., **kwargs):
        super(MultiplicativeGaussianNoise, self).__init__(**kwargs)
        self.std = std


    def apply(self, x):
        noise = self.rng_theano.normal(size=x.shape) * self.std
        return x * noise



class ConcatenatedGaussianNoise(NonDeterministicLayer):
    def __init__(self, channels, **kwargs):
        super(ConcatenatedGaussianNoise, self).__init__(**kwargs)
        self.channels = channels


    def apply(self, x):
        if x.ndim in [2, 3]:
            # bc or tbc
            i = x.ndim - 1
        else:
            # bc01 or tbc01
            i = x.ndim - 3

        pattern = [x.shape[j] for j in range(x.ndim)]
        pattern[i] = self.channels
        noise = self.rng_theano.normal(size=tuple(pattern))

        return T.concatenate([x, noise], axis=i)



class RandomCrop(NonDeterministicLayer):
    def __init__(self, shape, **kwargs):
        super(RandomCrop, self).__init__(**kwargs)
        self.shape = shape
        # need to use this stream for randint
        self.rng_stream = T.shared_randomstreams.RandomStreams(self.seed)


    def set_io_dims(self, tup):
        self.input_dims = tup
        self.output_dims = (tup[0],) + self.shape


    def fprop(self, x, **kwargs):
        # should the validation behavior be the middle crop?
        return self.apply(x)


    def apply(self, x):
        # collapse batch and channel for fancy indexing
        out = x.reshape((x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))

        rand_range = self.input_dims[-1] - self.output_dims[-1]

        x_offsets = self.rng_stream.random_integers(size=[out.shape[0]], high=rand_range, ndim=1, dtype='int32')
        y_offsets = self.rng_stream.random_integers(size=[out.shape[0]], high=rand_range, ndim=1, dtype='int32')

        out = out[T.arange(out.shape[0])[:,None,None],
                  x_offsets[:,None,None] + T.arange(self.shape[0])[:,None],
                  y_offsets[:,None,None] + T.arange(self.shape[0])]
        out = out.reshape((x.shape[0],x.shape[1],out.shape[1],out.shape[2]))

        return out


if __name__ == '__main__':
    import numpy as np
    x = T.ftensor4('x')
    lay = RandomCrop((34,34),input_dims=(3,45,45))
    lay.set_io_dims(lay.input_dims)
    y = lay.fprop(x)

    f = theano.function([x],[y])
    print f(np.random.random((10,3,45,45)).astype(np.float32))[0].shape
