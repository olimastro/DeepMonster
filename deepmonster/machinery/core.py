from dicttypes import dictify_type, frozendict, merge_dict_as_type
from linkers import LinksHolder, LinksHelper
from deepmonster.utils import assert_iterable_return_iterable


class configdict(frozendict):
    """The config dictionary carried around by Core objects should be frozen, so we
    don't have unexpected behaviors and maybe help a bit when errors are thrown.

    Config will typically be dict of dicts. We find ourselves always writing
    config['x']['y']..... to finally hit our last dict and value we wanted.
    __getitem__ is therefore modified to allow searches in a filesystem path like fashion
    so one can write config['x/y/.../z'] to go look for that buried 'z' value.

    This is especially useful for has_key.
    """
    def __getitem__(self, key):
        if not isinstance(key, str):
            return dict.__getitem__(self, key)
        keys = key.split('/')
        v = self
        fetched_keys = []
        for k in keys:
            try:
                # we've gone too deep
                if not isinstance(v, dict):
                    raise KeyError
                v = dict.__getitem__(v, k)
            except KeyError:
                if len(fetched_keys) == 0:
                    raise KeyError("Config query {} not found".format(k))
                raise KeyError(
                    "Trying to fetch {} after successful queries {}".format(k, fetched_keys) +\
                    " in a configuration dict.")
            fetched_keys.append(k)
        return v


    def has_key(self, key):
        if not isinstance(key, str):
            return dict.has_key(self, key)
        keys = key.split('/')
        v = self
        for i, k in enumerate(keys):
            if not isinstance(v, dict):
                return False
            elif i == len(keys) - 1:
                break
            elif dict.has_key(v, k):
                v = dict.__getitem__(v, k)
                continue
            return False
        return dict.has_key(v, keys[-1])



class Core(object):
    """Base class for a configurable object of the library. The rationale of this class
    is to bypass the normal __init__ method call as it is not flexible enough
    for the purposes of Core objects (or too flexible?).
    The only __init__ value for them is a dict config.
    ***ONLY modify __init__ with care when you know what you are doing***

    This way, Core objects can fetch wanted values through self.config.
    To emulate what user custom __init__ would do, a method configure() is
    called at the end and can be implemented for further customization.

    They also can depend on other core objects which are directly set to them
    with the 'bind_a_core' method. These dependancies need to be specified in
    __core_dependancies__.

    The config dict is of type frozendict so it cannot be easily changed. This is
    a design choice to try enforcing a bit of reproductibility. The user
    should know what happened in a DM run by inspection of the .yml files and
    hence the config. If objects start to modify freely these values and wildly
    change their behaviors accordingly, then inspection becomes useless. Core still
    provides a method to write on its own local config dict (it will not propagate
    to other Core's config).
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
        self.config = dictify_type(config, configdict)
        self.configure()


    def bind_a_core(self, core_id, core_obj):
        assert core_id in self.__core_dependancies__, \
                "{} does not depend on {}".format(self.__class__.__name__, core_id)
        setattr(self, core_id, core_obj)


    def configure(self):
        """Define actions on how to configure this object following its init.
        """
        pass


    def assert_for_keys(self, keys):
        """Check if config has the keys and throw an error if not.
        It is possible to look in a hiarachy fasion if the user know that a
        key map to a dict with other keys the user would like to check.

        Ex.: self.check_for_keys(['foo'/bar']) will in effect result in a
        call to self.config['foo'].has_key('bar')
        """
        keys = assert_iterable_return_iterable(keys, 'list')
        has_keys = map(self.config.has_key, keys)
        assert sum(has_keys) == len(keys), "Missing config keys {}".format(
            [x for x, y in zip(keys, has_keys) if not y])


    def write_on_config(self, newitem, from_key=None):
        """See above why this method exists compared on writing directly on self.config

        from_key specifies the root key on which to update since the config
        is a dict of dicts / non dicts. It can be a list.
        If unspecified, we write from the base of the dict.
        """
        config = dictify_type(self.config, dict) # unfreeze the config
        if from_key is not None:
            D = config
            from_key = assert_iterable_return_iterable(from_key, 'list')
            for k in from_key:
                D = D[k]
            D.update(newitem)
        else:
            config.update(newitem)

        self.config = dictify_type(config, configdict)


    def merge_config_with_priority_args(self, dict_of_args, config_subset=None):
        """In various occasions, objects could have available for themselves
        other dictionaries of config values than the global ones assigned to
        them in self.config.

        This method returns a prioritydict merged of config and dict_of_args.

        ***self.config can be arbitrarely large. If subset is specified, this
        will return all pairs in dict_of_args AND ONLY the remaining ones
        in self.config which are in subset.
        """
        subset = {k: v for k,v in self.config.iteritems() if k in config_subset} \
                if config_subset is not None else self.config
        return merge_dict_as_type(dict_of_args, subset, keep_source_type=False)



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

        holders = [x.linksholder for x in cores if hasattr(x, 'linksholder')]
        return LinksHelper.merge_holders(holders)


    def store_links(self, links):
        self.linksholder.store_links(links)
