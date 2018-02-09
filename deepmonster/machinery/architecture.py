from dicttypes import onlynewkeysdict
from core import LinkingCore
from linkers import ModelParametersLink
from deepmonster.nnet.network import Feedforward
from deepmonster.utils import flatten

def build_group(group):
    """Architecture decorator a build method. It is usually very handy to be able
    to access a whole group of architectures and form a superset.
    Apply this decorator to an Architecture
    method so it binds every add_layers_to_arch call to a group
    """
    def group_decorator(func, *args, **kwargs):
        def set_group(*args, **kwargs):
            arch = args[0]
            arch.is_building_group = group
            rval = func(*args, **kwargs)
            arch.is_building_group = None
            return
        return set_group
    return group_decorator


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
        self._is_building_group = None
        self._group = {}
        self.arch_name = self.config.get('name', '')
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
        params = [x.params for x in self._internal.values() if \
                  hasattr(x, 'params')]
        return flatten(params)

    @property
    def is_building_group(self):
        return self._is_building_group

    @is_building_group.setter
    def is_building_group(self, value):
        if value is None:
            self._is_building_group = None
        else:
            self._is_building_group = value
            self._group.update({value: []})

    def add_to_group(self, member):
        if self.is_building_group is None:
            return
        self._group[self.is_building_group].append(member)

    def get_arch_group(self, group):
        return self._group[group]


    def add_layers_to_arch(self, layers, prefix, ff_kwargs):
        """Helper method to easily build Feedforward object and add
        it in the internal dict.

        Wrap any kwargs to give to Feedforward in ff_kwargs dict.
        """
        ff = Feedforward(layers, prefix, **ff_kwargs)
        self._internal.update({prefix: ff})
        self.add_to_group(prefix)


    def add_to_arch(self, key, value):
        """Set directly the internal. This is a design choice, the user
        could just do arch._internal[...] = ..., it is to emphasise
        that _internal is not to be handled directly.
        """
        self._internal.update({key: value})
        self.add_to_group(key)

    def restrict_architecture(self):
        """Cast _internal in as a onlynewkeysdict type. This will make
        it throw errors if trying to add keys that already exist and
        could help debugging
        """
        self._internal = onlynewkeysdict(self._internal)

    def build_arch(self):
        raise NotImplementedError("Base class does not implement an arch")
