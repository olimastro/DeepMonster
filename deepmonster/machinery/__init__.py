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
        assert issubclass(config.__class__, dict), "Configuration file needs to \
                be of type dict"
        for k,v in kwargs.iteritems():
            self.bind_a_core(k, v)
        self.config = frozendict(config)
        self.configure()


    def bind_a_core(self, core_id, core_obj):
        assert core_id in self.__core_dependancies__, \
                "{} does not depend on {}".format(self.__name__, core_id)
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



class frozendict(dict):
    def update(self, otherdict):
        raise TypeError("Cannot update a frozen dict")

    def __setitem__(self, k, v):
        raise TypeError("Cannot setitem on a frozen dict")


class prioritydict(dict):
    def get_only_new_items(self, otherdict):
        only_new_keys = set(otherdict.keys()) - set(self.keys())
        newdict = {k: v for k,v in otherdict.iteritems() if k in only_new_keys}
        return newdict

    def update(self, otherdict):
        newdict = self.get_only_new_items(otherdict)
        super(prioritydict, self).update(newdict)

    def __setitem__(self, k, v):
        newdict = self.get_only_new_items({k: v})
        if len(newdict) == 0:
            return
        super(prioritydict, self).__setitem__(k, v)


class onlynewkeysdict(dict):
    def error_on_existing_key(self, keys):
        bad_keys = filter(self.has_key, keys)
        if len(bad_keys) > 0:
            raise TypeError("Cannot set / update value of existing key of \
                            onlynewkeysdict type of dict. Key(s) causing \
                            errors: {}".format(bad_keys))

    def update(self, otherdict):
        self.error_on_existing_key(otherdict.keys())
        super(onlynewkeysdict, self).update(otherdict)

    def __setitem__(self, k, v):
        self.error_on_existing_key([k])
        super(onlynewkeysdict, self).__setitem__(k, v)
