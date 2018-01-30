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


def dictify_type(d, dtype):
    """Propagate dtype to all dicts (on all levels) of d
    """
    # casting seems to make new objects and we lose the references in dicts of dicts
    d = dict(d) # to prevent errors from update we need to recast as dict
    for k,v in d.iteritems():
        if issubclass(v.__class__, dict):
            v = dictify_type(v, dtype)
            d[k] = v

    return dtype(d)


# DEFAULT library execution when merging dictionaries through these func
MERGETYPE = prioritydict


def merge_dict_as_type(d_source, d_tomerge, mergetype=MERGETYPE, keep_source_type=True):
    """Return a new dict that is d_source updated with an update policy as specified
    in mergetype with d_tomerge. By default, it uses a prioritydict.

    If keep_source_type returns a dict of the same type as input else returns
    a dict of mergetype
    """
    new_dict = mergetype(d_source)
    new_dict.update(d_tomerge)
    if keep_source_type:
        return type(d_source)(new_dict)
    return new_dict


def merge_all_dicts_with_dict(dict_of_dicts, dict_to_merge, mergetype=MERGETYPE):
    """Merge all dicts in dict of dicts with dict_to_merge with a merging behavior
    according to MERGETYPE
    """
    for k, v in dict_of_dicts.iteritems():
        dict_of_dicts[k] = merge_dict_as_type(v, dict_to_merge, mergetype)

    return dict_of_dicts
