from deepmonster.machinery import Architecture
from deepmonster.nnet.simple import FullyConnectedLayer
from deepmonster.nnet.activations import LeakyRectifier, Identity
from deepmonster.nnet.initializations import Initialization, Gaussian

class WrongArch(Architecture):
    pass

class DummyArch(Architecture):
    def build_arch(self):
        self.restrict_architecture()

        layers = [
            FullyConnectedLayer(input_dims=self.config['input_dims'],
                                output_dims=128),
            FullyConnectedLayer(output_dims=1, activation=Identity())
        ]

        arch_config = {
            'batch_norm': True,
            'activation': LeakyRectifier(leak=self.config['leak']),
            'initialization': Initialization({'W': Gaussian(std=0.5)})
        }

        self.add_layers_to_arch(layers, 'fully', arch_config)

