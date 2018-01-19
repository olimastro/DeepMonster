from dicttypes import onlynewkeysdict
from core import LinkingCore
from linkers import ModelParametersLink
from deepmonster.nnet.network import Feedforward
from deepmonster.utils import flatten

class Architecture(LinkingCore):
    """Architecture class to build and store neural networks.
    build_arch should be overriden by the user to implement the user's arch.
    You can modify directly the _internal dict but Architecture provides
    add_layers_to_arch and add_to_arch methods to help the process.

    Provides all the same getters as a regular dict, but setters are more
    restricted.
    """
    def configure(self):
        self._internal = {}
        self.arch_name = ''
        self.build_arch()
        mpl = ModelParametersLink(self.parameters, name=self.arch_name)
        self.linksholder.store_links(mpl)

    def __len__(self):
        return len(self._internal)

    def __getitem__(self, key):
        return self._internal[key]

    def keys(self):
        return self._internal.keys()

    def has_key(self, key):
        return self._internal.has_key(key)

    @property
    def parameters(self):
        params = [x.parameters for x in self._internal.values() if \
                  hasattr(x, 'parameters')]
        return flatten(params)

    def add_layers_to_arch(self, layers, prefix, ff_kwargs):
        """Helper method to easily build Feedforward object and add
        it in the internal dict.

        Wrap any kwargs to give to Feedforward in ff_kwargs dict.
        """
        ff = Feedforward(layers, prefix, **ff_kwargs)
        self._internal.update({prefix: ff})

    def add_to_arch(self, key, value):
        """Set directly the internal. This is a design choice, the user
        could just do arch._internal[...] = ..., it is to emphasise
        that _internal is not to be handled directly.
        """
        self._internal.update({key: value})

    def restrict_architecture(self):
        """Cast _internal in as a onlynewkeysdict type. This will make
        it throw errors if trying to add keys that already exist and
        could help debugging
        """
        self._internal = onlynewkeysdict(self._internal)

    def build_arch(self):
        raise NotImplementedError("Base class does not implement an arch")
