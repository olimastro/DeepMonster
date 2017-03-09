import numpy as np
import theano
import theano.tensor as T
from baselayers import Layer


class FullyConnectedLayer(Layer) :
    def param_dict_initialization(self):
        dict_of_init = {
            'W' : [(self.input_dims[0],self.output_dims[0],), 'norm', 0.1]}

        if self.use_bias or self.batch_norm:
            dict_of_init.update({
            'betas' : [self.output_dims[0], 'zeros'],
            })

        self.param_dict = dict_of_init


    def apply(self, x):
        if x.ndim == 4:
            x = x.transpose(0,2,3,1)
        elif x.ndim == 2:
            pass
        else:
            raise ValueError("Where are you going with these dimensions in a fullyconnected?")

        out = T.dot(x, self.W)
        if x.ndim == 4:
            out = out.transpose(0,3,1,2)

        return out



class FullyConnectedOnLastTime(FullyConnectedLayer):
    """
        Applies a fully connected layer on the last output of a 5D tensor.
        Most likely used for the last time step.
    """
    def apply(self, x):
        return super(FullyConnectedOnLastTime, self).apply(x[-1])



class NoiseConditionalLayer(FullyConnectedLayer):
    """
        This class takes x and apply a linear transform on it to expend it
        into mu, sigma and return mu + sigma * noise
    """
    def __init__(self, noise_type='gaussian', **kwargs):
        assert noise_type in ['gaussian', 'uniform']
        super(NoiseConditionalLayer, self).__init__(**kwargs)
        self.noise_type = noise_type
        self.activation = None


    def param_dict_initialization(self):
        dict_of_init = {
            'W' : [(self.input_dims[0],self.output_dims[0] * 2,), 'norm', 0.1]}

        if self.use_bias:
            dict_of_init.update({
            'betas' : [(self.output_dims[0] * 2,), 'zeros'],
            })
        self.param_dict = dict_of_init


    def fprop(self, x, **kwargs):
        mulogsigma = super(NoiseConditionalLayer, self).fprop(x, **kwargs)
        det = kwargs.get('deterministic', False)

        slice_dim = self.output_dims[0]
        if x.ndim in [2, 4]:
            mu = mulogsigma[:,:slice_dim]
            logsigma = mulogsigma[:,slice_dim:]
        elif x.ndim in [3, 5]:
            mu = mulogsigma[:,:,:slice_dim]
            logsigma = mulogsigma[:,:,slice_dim:]

        if det:
            epsilon = 1.
        else:
            if self.noise_type == 'gaussian':
                epsilon = self.rng_theano.normal(size=mu.shape)
            elif self.noise_type == 'uniform':
                epsilon = self.rng_theano.uniform(size=mu.shape)

        noised_x = mu + sigma * epsilon

        return noised_x



if __name__ == '__main__' :
    from activations import Rectifier
    from convolution import *
    fl = FullyConnectedLayer(input_size=3,output_size=97,
                             prefix='fl',weight_norm=False,
                             batch_norm=True,gamma_scale=0.1,
                             activation=Rectifier())
    fl.initialize()

    dtensor5 = T.TensorType('float32', (False,)*5)
    #x = dtensor5('x')
    theano.config.compute_test_value = 'warn'
    x = T.ftensor4('x')
    npx = np.random.random((50, 3, 1, 1)).astype(np.float32)
    x.tag.test_value = npx

    y = fl.fprop(x)

    f = theano.function([x],[y])
    out = f(npx)

    bn_updates = [(fl.pop_mu, fl.mean)]
    fun = theano.function([x], [], updates=bn_updates)
    for i in range(10):
        fun(npx)
    fl.bn_flag.set_value(0)
    out = f(npx)
    fl.bn_flag.set_value(1)
    out = f(npx)
    import ipdb ; ipdb.set_trace()
