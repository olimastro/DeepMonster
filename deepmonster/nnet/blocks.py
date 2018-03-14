from convolution import ConvLayer, Conv3DLayer
from network import StandardBlock
from simple import FullyConnectedLayer, FullyConnectedOnLastTime
from wrapped import ConvLayer5D


class ConvBlock(StandardBlock):
    apply_layer_type = {
        'default': ConvLayer,
        'wrap_time': ConvLayer5D}

class Conv3DBlock(StandardBlock):
    apply_layer_type = Conv3DLayer

class FullyConnectedBlock(StandardBlock):
    apply_layer_type = {
        'default': FullyConnectedLayer,
        'on_last_time': FullyConnectedOnLastTime}
