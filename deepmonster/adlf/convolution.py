import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nnet as nnet # for new backend
from theano.tensor.nnet.abstract_conv import AbstractConv2d_gradInputs

import utils
from baselayers import Layer


class ConvLayer(Layer) :
    def __init__(self, filter_size, num_filters, strides=(1,1), padding='valid',
                 tied_bias=True, image_size=None, num_channels=None,
                 **kwargs):
        super(ConvLayer, self).__init__(**kwargs)
        image_size = utils.parse_tuple(image_size, 2)
        self.image_size = image_size

        self.tied_bias = tied_bias
        self.num_filters = num_filters
        self.strides = utils.parse_tuple(strides, 2)
        if not isinstance(padding, str):
            padding = utils.parse_tuple(padding, 2)
        self.padding = padding

        self.num_channels = num_channels
        if image_size is None:
            self.input_dims = (num_channels, None, None)
        else:
            self.input_dims = (num_channels, image_size[0], image_size[1])

        self.filter_size = utils.parse_tuple(filter_size, 2)


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
            raise TypeError("Does not recognize padding type {} in {}".format(self.padding,self.prefix))
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
        return nnet.conv2d(x, W, subsample=strides, border_mode=border_mode)


    def param_dict_initialization(self):
        if self.tied_bias :
            biases_dim = (self.num_filters)
        else :
            biases_dim = self.output_dims

        dict_of_init = {
            'W' : [(self.num_filters, self.num_channels)+self.filter_size,
                   'norm', 0.1]}

        if self.use_bias or self.batch_norm :
            dict_of_init.update({
                'betas' : [biases_dim, 'zeros'],
        })
        self.param_dict = dict_of_init


    def apply_bias(self, x):
        if self.tied_bias:
            return super(ConvLayer, self).apply_bias(x)
        else:
            pattern = ('x',) * (x.ndim - 3) + (0,1,2)
            return x + self.betas.dimshuffle(*pattern)


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



# Utilities classes to enable convlayers on a 5D tensors by surrounding their
# fprop by reshapes. Collapse batch and time axis together

class ConvLayer5D(ConvLayer):
    def fprop(self, x, **kwargs):
        if x.ndim == 5:
            y = x.reshape((x.shape[0]*x.shape[1],x.shape[2],x.shape[3],x.shape[4]))
            out = super(ConvLayer5D, self).fprop(y, **kwargs)
            out = out.reshape((x.shape[0],x.shape[1],out.shape[1],out.shape[2],out.shape[3]))
        else:
            # act normal
            out = super(ConvLayer5D, self).fprop(x, **kwargs)
        return out


# according to the MRO the only thing it inherits from ConvLayer5D is the fprop
class DeConvLayer5D(ConvLayer5D, DeConvLayer):
    pass
