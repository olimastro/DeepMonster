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
