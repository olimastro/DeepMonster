from dicttypes import frozendict, prioritydict
from linkers import LinksHolder

class configdict(frozendict):
    """The config dictionary carried around by Core objects should be frozen, so we
    don't have unexpected behaviors and maybe help a bit when errors are thrown.
    """
    def __getitem__(self, key):
        # does this try/except slows down everything?
        try:
            return super(configdict, self).__getitem__(key)
        except KeyError:
            raise KeyError("Configuration query error, `{}` cannot be found.".format(key))


class Core(object):
    """Base class for a configurable object of the library. The rationale of this class
    is to bypass the normal __init__ method call as it is not flexible enough
    for the purposes of Core objects (or too flexible?).
    The only __init__ value for them is a dict config.
    ***ONLY modify __init__ with care when you know what you are doing***

    Therefore, Core objects can fetch wanted values through self.config.
    To emulate what user custom __init__ would do, a method configure() is
    called at the end and can be implemented for further customization.

    They also can depend on other core objects which are directly set to them
    with the 'bind_a_core' method. These dependancies need to be specified in
    __core_dependancies__.

    Because all Core objects will most likely share the same config dict (since
    they are global values setting the global behaviors of the experiment
    by the user), config is frozen and cannot be changed after its parsing.
    """
    __core_dependancies__ = []

    def __init__(self, config, **kwargs):
        # kwargs can be a dict of cores to bind to this object. It is provided
        # in __init__ to ease usage of Core object not in the standard way.
        #import ipdb; ipdb.set_trace()
        assert issubclass(config.__class__, dict), "Configuration file needs to" +\
                " be of type dict."
        for k,v in kwargs.iteritems():
            self.bind_a_core(k, v)
        self.config = configdict(config)
        self.configure()


    def bind_a_core(self, core_id, core_obj):
        assert core_id in self.__core_dependancies__, \
                "{} does not depend on {}".format(self.__class__.__name__, core_id)
        setattr(self, core_id, core_obj)


    def configure(self):
        """Define actions on how to configure this object following its init.
        """
        pass


    def merge_config_with_priority_args(dict_of_args, config_subset=None):
        """In various occasions, objects could have available for themselves
        other dictionaries of config values than the global ones assigned to
        them in self.config.

        This method returns a prioritydict merged of config and dict_of_args.

        ***self.config can be arbitrarely large. If subset is specified, this
        will return all pairs in dict_of_args AND ONLY the remaining ones
        in self.config which are in subset.
        """
        new_dict = prioritydict(dict_of_args)
        subset = {k: v for k,v in self.config.iteritems() if k in config_subset} \
                if config_subset is None else self.config
        return new_dict.update(subset)



class LinkingCore(Core):
    """All Core objects used in the library when hatching a monster are of this class.
    This class provides methods and an interface to manipulate linkers held by the cores.
    """
    def __init__(self, *args, **kwargs):
        self.linksholder = LinksHolder()
        super(LinkingCore, self).__init__(*args, **kwargs)

    def combine_holders(self, on_cores='all'):
        """Return requested combination of holders
        """
        if on_cores == 'all':
            on_cores = self.__core_dependancies__
        cores = [getattr(self, c) for c in self.__core_dependancies__ \
                 if c in on_cores]

        return sum(map(lambda x: x.linksholder, cores))
