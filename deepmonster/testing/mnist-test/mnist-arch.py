from deepmonster.machinery.architecture import Architecture
from deepmonster.nnet.activations import Rectifier, Softmax
from deepmonster.nnet.convolution import ConvLayer
from deepmonster.nnet.initializations import Initialization, Gaussian
from deepmonster.nnet.extras import Reshape
from deepmonster.nnet.simple import FullyConnectedLayer


class MnistArch(Architecture):
    def configure(self):
        self.assert_for_keys(['image_size', 'channels'])
        self.image_size = self.config['image_size']
        self.channels = self.config['channels']
        super(MnistArch, self).configure()


    def build_arch(self):
        self.restrict_architecture()

        config = {
            'activation': Rectifier(),
            'batch_norm': True,
            'weight_norm': False,
            'initialization': Initialization({'W': Gaussian(std=0.03)}),
            'padding': 'half',
        }

        layers = [
            ConvLayer(3, 16, image_size=self.image_size, num_channels=self.channels),
            ConvLayer(3, 32, strides=(2,2)), # 14x14
            ConvLayer(3, 64, strides=(2,2)), # 7x7
            ConvLayer(4, 128, padding='valid'), # 4x4
            ConvLayer(4, 256, padding='valid'), # 1x1
            Reshape(([0], -1)),
            FullyConnectedLayer(input_dims=256, output_dims=10,
                                activation=Softmax(), batch_norm=False)
        ]
        self.add_layers_to_arch(layers, 'classifier', config)
