import numpy as np
import os
import theano
import theano.tensor as T

from deepmonster.adlf.activations import Rectifier
from deepmonster.adlf.initializations import Initialization, Gaussian, Orthogonal
from deepmonster.adlf.network import Feedforward, find_nets, find_attributes
from deepmonster.adlf.rnn import ConvLSTM, LSTM
from deepmonster.adlf.utils import (getftensor5, collapse_time_on_batch,getnumpyf32,
                                    expand_time_from_batch, stack_time)

image_size = (120,120)
batch_size = 32
channels = 1

config = {
    'batch_norm' : True,
    'activation' : Rectifier(),
    'initialization' : Initialization({'W' : Gaussian(std=0.0333),
                                       'U' : Orthogonal(0.0333)}),
}
# REMINDER: If a config is set directly on a layer, it will have priority!
# REMINDER2: CONVLSTM always have Tanh() as non-linearity

layers = [
    #ConvLSTM(5, 64, strides=(2,2), padding='half',
    #         num_channels=channels, image_size=image_size), # 60x60
    #ConvLayer(5, 64, strides=(2,2), padding='half',
    #          num_channels=channels, image_size=image_size), # 60x60
    #ConvLayer(3, 64, strides=(1,1), padding='half'),
    #ConvLayer(3, 128, strides=(2,2), padding='half'), # 30x30
    #ConvLSTM(3, 128, strides=(1,1), padding='half'),
    LSTM(input_dims=50, output_dims=200),
]
net = Feedforward(layers, 'net', **config)

#x = getftensor5()('x')
x = T.ftensor3('x')
y = net.fprop(x**2 / 2.)
cost = y.mean()

parameters = net.params

from blocks.algorithms import Scale
from blocks.algorithms import GradientDescent
optimizer = Scale(0.)

print "Calling Algorithm"
algorithm = GradientDescent(
    #gradients=grads, parameters=parameters,
    cost=cost,
    parameters=parameters,
    step_rule=optimizer)

from theano.compile.nanguardmode import NanGuardMode
fun = theano.function(
    inputs=[x],outputs=[cost],updates=algorithm.updates,
    mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
#npx = getnumpyf32((5, batch_size, channels,)+image_size)
npx = np.random.random((5,32,50)).astype(np.float32)
out = fun(npx)
#for i,v in enumerate(parameters):
#    if 'U' in v.name:
#        theano.printing.debugprint(algorithm.updates[i][1])
#        break
