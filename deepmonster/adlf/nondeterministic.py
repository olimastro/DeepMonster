import theano
import theano.tensor as T

from baselayers import RandomLayer


class NonDeterministicLayer(RandomLayer):
    """
        Any layer subclassing this class should have limited applications
        such as dropout where it changes the theano graph only for
        deterministic == False
    """
    def fprop(self, x, **kwargs):
        det = kwargs.pop('deterministic', False)
        if det:
            return x
        else:
            return self.apply(x)



class Dropout(NonDeterministicLayer):
    def __init__(self, p, mode='normal', **kwargs):
        super(Dropout, self).__init__(**kwargs)
        assert mode in ['normal', 'spatial']
        self.p = 1.- p
        self.mode = mode


    def apply(self, x):
        if self.mode is 'normal' or x.ndim == 2:
            shape = x.shape
            dropout_mask = self.rng_theano.binomial(x.shape, p=self.p, dtype='float32') / self.p
        elif x.ndim == 4 or x.ndim == 5:
            # spatial dropout, meaning drop a whole feature map
            shape = (x.shape[x.ndim-3],)
            pattern = ('x',) * (x.ndim-3) + (0,) + ('x','x',)

            dropout_feature_mask = self.rng_theano.binomial(shape, p=self.p, dtype='float32') / self.p
            dropout_mask = T.ones_like(x) * dropout_feature_mask.dimshuffle(*pattern) / self.p
        return x



class AdditiveGaussianNoise(NonDeterministicLayer):
    def __init__(self, std, **kwargs):
        super(AdditiveGaussianNoise, self).__init__(**kwargs)
        self.std = std


    def apply(self, x):
        noise = self.rng_theano.normal(size=x.shape) * self.std
        return x + noise



class MultiplicativeGaussianNoise(NonDeterministicLayer):
    def __init__(self, std, **kwargs):
        super(MultiplicativeGaussianNoise, self).__init__(**kwargs)
        self.std = std


    def apply(self, x):
        noise = self.rng_theano.normal(size=x.shape) * self.std
        return x * noise
