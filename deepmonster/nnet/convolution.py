import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet.abstract_conv import AbstractConv2d_gradInputs
from theano.gpuarray.basic_ops import gpu_contiguous

from baselayers import AbsLayer, ParametrizedLayer
from utils import parse_tuple


class UntiedBiasLayer(ParametrizedLayer):
    @property
    def param_dict_initialization(self):
        dict_of_init = {
            'betas' : [self.output_dims, 'zeros'],
        }
        return dict_of_init


    def apply(self, x):
        pattern = ('x',) * (x.ndim - 3) + (0,1,2)
        return x + self.betas.dimshuffle(*pattern)


class ConvLayer(ParametrizedLayer) :
    def __init__(self, filter_size, num_filters, strides=(1,1), padding=None,
                 image_size=None, num_channels=None, **kwargs):
        super(ConvLayer, self).__init__(**kwargs)
        image_size = parse_tuple(image_size, 2)
        self.image_size = image_size

        self.num_filters = num_filters
        self.strides = parse_tuple(strides, 2)
        if isinstance(padding, int):
            padding = parse_tuple(padding, 2)
        self.padding = padding

        self.num_channels = num_channels
        if image_size is None:
            self.input_dims = (num_channels, None, None)
        else:
            self.input_dims = (num_channels, image_size[0], image_size[1])

        self.filter_size = parse_tuple(filter_size, 2)


    def infer_outputdim(self):
        i_dim = self.image_size[0]
        k_dim = self.filter_size[0]
        s_dim = self.strides[0]
        border_mode = self.padding
        if border_mode == 'valid' :
            border_mode = 0
        elif border_mode == 'half' :
            border_mode = k_dim // 2
        elif border_mode == 'full':
            border_mode = k_dim - 1
        elif isinstance(border_mode, tuple):
            border_mode = border_mode[0]
        else:
            raise ValueError("Does not recognize padding {} in {}".format(
                self.padding, self.prefix))
        self._border_mode = parse_tuple(border_mode, 2)
        o_dim = (i_dim + 2 * border_mode - k_dim) // s_dim + 1

        self.feature_size = (o_dim, o_dim)


    def set_io_dims(self, tup):
        if self.num_channels is None:
            self.num_channels = tup[0]
        if self.image_size == (None,None):
            self.image_size = tup[1:]
        self.input_dims = tup
        self.infer_outputdim()
        self.output_dims = (self.num_filters,) + self.feature_size


    def convolve(self, x, W, strides, border_mode):
        return T.nnet.conv2d(x, W, subsample=strides, border_mode=border_mode)


    @property
    def param_dict_initialization(self):
        dict_of_init = {
            'W' : [(self.num_filters, self.num_channels)+self.filter_size,
                   'norm', 0.1]}
        return dict_of_init


    def apply(self, x):
        if self.filter_size == (1,1) and self.input_dims[1:] == (1,1):
            out = T.dot(x.flatten(2), self.W.flatten(2).dimshuffle(1,0))
            out = out[:,:,None,None]
        else :
            out = self.convolve(x, self.W, self.strides, self.padding)

        return out



class DeConvLayer(ConvLayer) :
    # ripped from blocks.conv.ConvolutionalTranspose.original_image_size
    def infer_outputdim(self):
        unused_edge = (0,0)

        if self.padding == 'full':
            border = tuple(k - 1 for k in self.filter_size)
        elif self.padding == 'half':
            border = tuple(k // 2 for k in self.filter_size)
        elif self.padding == 'valid':
            border = [0] * len(self.image_size)
        else :
            border = self.padding
        tups = zip(self.image_size, self.strides, self.filter_size, border,
                   unused_edge)

        out = tuple(s * (i - 1) + k - 2 * p + u for i, s, k, p, u in tups)

        self.feature_size = out

        return out


    # make it compatible with the blocks copy pasta
    def _get_outdim(self):
        return (self.num_filters,) + self.infer_outputdim()


    # ripped from blocks.conv.ConvolutionalTranspose.conv2d_impl
    def convolve(self, input_, W, strides, border_mode):
        # The AbstractConv2d_gradInputs op takes a kernel that was used for the
        # **convolution**. We therefore have to invert num_channels and
        # num_filters for W.
        W = W.transpose(1, 0, 2, 3)
        imshp = (None,) + self._get_outdim()
        #kshp = (filter_shape[1], filter_shape[0]) + filter_shape[2:]
        kshp = (self.num_channels, self.num_filters,) + self.filter_size
        #import ipdb; ipdb.set_trace()
        return AbstractConv2d_gradInputs(
            imshp=imshp, kshp=kshp, border_mode=border_mode,
            subsample=strides)(W, input_, self._get_outdim()[1:])



class Conv3DLayer(ConvLayer) :
    """
        This class could be easy, but nothing is easy in life.
        The theano op is implemented in bct01 and this whole library has been
        written in tbc01. The major work of this class will be to accomodate
        for this annoyance.

        If dimshuffle_inp == True: assume the input is in tbc01 and we have to
        shuffle it.
        If dimshuffle_inp == False: assume it comes in bct01.
        If using a stack of Conv3D it is better to only dimshuffle twice
        at the io of the stack for memory and comp usage.
    """
    def __init__(self, filter_size, num_filters, dimshuffle_inp=True,
                 pad_time=(0,), **kwargs):
        # a bit of fooling around to use ConvLayer.__init__
        time_filter_size, filter_size = self._seperate_time_from_spatial(filter_size)
        strides = kwargs.pop('strides', (1,1,1))
        time_stride, strides = self._seperate_time_from_spatial(strides)
        super(Conv3DLayer, self).__init__(filter_size, num_filters, strides=strides, **kwargs)
        self.time_filter_size = time_filter_size
        self.time_stride = time_stride
        self.dimshuffle_inp = dimshuffle_inp
        self.pad_time = parse_tuple(pad_time)
        self._gemm = False


    def force_gemm(self):
        self._gemm = True


    def _seperate_time_from_spatial(self, tup):
        if isinstance(tup, tuple):
            time = tup[0]
            space = tup[1:]
        else:
            time = tup
            space = parse_tuple(tup, 2)
        return (time,), space


    @property
    def param_dict_initialization(self):
        dict_of_init = {
            'W' : [(self.num_filters, self.num_channels) + \
                   self.time_filter_size + self.filter_size,
                   'norm', 0.1]}
        return dict_of_init


    def apply(self, x):
        subsample = self.time_stride + self.strides
        border_mode = self.pad_time + self._border_mode

        # sometimes optim fails to use dnn, so fallback to gemm
        if self._gemm:
            out = theano.gpuarray.blas.GpuCorr3dMM(
                border_mode=border_mode, subsample=subsample)(gpu_contiguous(x), gpu_contiguous(self.W))
        else:
            out = T.nnet.conv3d(x, self.W,
                                border_mode=border_mode,
                                subsample=subsample,
                                filter_flip=False)
        return out


    def fprop(self, x, **kwargs):
        # x incoming is in tbc01
        # all this class assumes theano's bct01
        if self.dimshuffle_inp:
            x = x.dimshuffle(1,2,0,3,4)
        out = super(Conv3DLayer, self).fprop(x, **kwargs)
        if self.dimshuffle_inp:
            out = out.dimshuffle(2,0,1,3,4)
        return out



class MaxPooling(AbsLayer):
    """Downsample by // 2 image size
    """
    def set_io_dims(self, tup):
        assert len(tup) == 3, "dims should be image dims (c01)"
        self.input_dims = tup
        self.output_dims = (tup[0], tup[1] // 2, tup[2] // 2,)
        assert all(i > 0 for i in tup), "downsampling below size 1"


    def apply(self, x):
        return T.signal.pool.pool_2d(x, ws=(2,2), ignore_border=True)
