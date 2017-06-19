import numpy as np
import scipy.stats as st

import theano
import theano.tensor as T
from theano.tensor.nnet.abstract_conv import bilinear_upsampling

from baselayers import AbsLayer


class BilinearUpsampling(AbsLayer):
    def __init__(self, ratio, use_1D_kernel=True, **kwargs):
        super(BilinearUpsampling, self).__init__(**kwargs)
        self.ratio = ratio
        self.use_1D_kernel = use_1D_kernel

    def set_io_dims(self, tup):
        self.input_dims = tup
        output_inner_dims = tuple(d * self.ratio for d in self.input_dims[-2:])
        self.output_dims = self.input_dims[:-2] + output_inner_dims
        #print "setting output dims", self.output_dims

    def apply(self, x):
        if x.ndim == 5 :
            out = x.reshape((x.shape[0]*x.shape[1],x.shape[2],x.shape[3],x.shape[4]))
        else:
            out = x

        out = bilinear_upsampling(out, ratio=self.ratio,
                                  use_1D_kernel=self.use_1D_kernel)
        if x.ndim == 5 :
            out = out.reshape((x.shape[0],x.shape[1],x.shape[2],out.shape[-2],out.shape[-1]))
        return out


class GaussianKernelUpsampling(AbsLayer):
    """
        This applies a gaussian kernel to blurr in the pixels value into the upsampled image
        All credits to Olexa (and opencv)
    """
    def __init__(self, ratio=2, kernlen=5, **kwargs):
        # I imagine the idea could be aplied to a bigger ratio
        assert ratio == 2
        # the kernel is stripped from opencv, need a function that output such a sharp gauss kern
        assert kernlen == 5
        self.ratio = ratio
        #kernel = T.as_tensor_variable(gkern(kernlen))
        #self.kernel = kernel.dimshuffle('x','x',0,1)
        kernel = np.asarray([[1,4,6,4,1],
                             [4,16,26,16,4],
                             [6,26,64,26,6],
                             [1,4,6,4,1],
                             [4,16,26,16,4]])
        kernel = kernel[None,None,:,:] / 64.
        self.kernel = T.as_tensor_variable(kernel.astype(np.float32))
        super(GaussianKernelUpsampling, self).__init__(**kwargs)

    def set_io_dims(self, tup):
        self.input_dims = tup
        output_inner_dims = tuple(d * self.ratio for d in self.input_dims[-2:])
        self.output_dims = self.input_dims[:-2] + output_inner_dims

    def apply(self, x):
        # first fill zeros between each pixels
        # assumes the last 3 are c01
        shape = tuple(x.shape[i] for i in range(x.ndim-2))
        out = T.zeros(shape + (x.shape[-2] * 2, x.shape[-1] * 2), dtype=x.dtype)
        out = T.inc_subtensor(out[...,::2,::2], x)

        # blurr in with gauss kernel conv
        if x.ndim == 5 :
            out = out.reshape((x.shape[0]*x.shape[1],x.shape[2],out.shape[-2],out.shape[-1]))

        # this is necessary to avoid cross channels shenanigans
        preconv = out.reshape((out.shape[0]*out.shape[1],1,out.shape[2],out.shape[3]))
        conved = T.nnet.conv2d(preconv, self.kernel, subsample=(1,1), border_mode='half')
        out = conved.reshape(out.shape)

        if x.ndim == 5 :
            out = out.reshape((x.shape[0],x.shape[1],x.shape[2],out.shape[-2],out.shape[-1]))
        return out


# not sharp enough
def gkern(kernlen=5, nsig=1):
        """Returns a 2D Gaussian kernel array."""
        interval = (2*nsig+1.)/(kernlen)
        x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw/kernel_raw.sum()
        return kernel.astype(np.float32)


if __name__ == '__main__':
    from utils import getftensor5
    npx = np.random.random((5,12,3,12,12)).astype(np.float32)
    ftensor5 = getftensor5()
    x = ftensor5('x')

    lay = GaussianKernelUpsampling()
    z = lay.fprop(x)

    f = theano.function([x],[z])
    y = f(npx)
    import ipdb; ipdb.set_trace()
