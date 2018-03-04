from convolution import ConvLayer, Conv3DLayer
from network import StandardBlock
from simple import FullyConnectedLayer, FullyConnectedOnLastTime


class ConvBlock(StandardBlock):
    apply_layer_type = ConvLayer

class Conv3DBlock(StandardBlock):
    apply_layer_type = Conv3DLayer

class FullyConnectedBlock(StandardBlock):
    apply_layer_type = FullyConnectedLayer
