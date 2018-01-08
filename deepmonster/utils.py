import re

def assert_iterable_return_iterable(x, iter_type='tup'):
    """Make sure that x is the iterable of said type
    Watch out for the differences among the three,
    ex.: if x=[3,3] => set(x)={3}

    By default it will tuplify x

    ***The term assert could be misleading as a typical use of this
    util function is to MAKE OUT an iterable out of a single
    object that is not for example. It will not crash if it
    is not iterable.
    """
    assert iter_type in ['tup', 'list', 'set', 'keep']
    iterable = {
        'tup': tuple,
        'list': list,
        'set': set,
    }
    if not isinstance(x, (list, tuple, set)):
        # the dummy wrapping in a tuple is necessary as those will
        # actually try to iterate on the object x, which at this point we've
        # found not to be iterable
        if iter_type == 'keep':
            raise TypeError("Cannot keep iterable type as it is not a simple iterable")
        return iterable[iter_type]((x,))
    if iter_type == 'keep':
        return x
    return iterable[iter_type](x)


def sort_by_numbers_in_file_name(list_of_file_names):
    def tryint(s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(s):
        """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
        """
        return [tryint(c) for c in re.split('(\-?[0-9]+)', s)]

    def sort_nicely(l):
        """ Sort the given list in the way that humans expect.
        """
        l.sort(key=alphanum_key)
        return l

    return sort_nicely(list_of_file_names)


def flatten(x):
    # flatten list made of tuple or list
    def _flatten(container):
        for i in container:
            if isinstance(i, (list, tuple)):
                for j in flatten(i):
                    yield j
            else:
                yield i
    return list(_flatten(x))


def make_dict_cls_name_obj(L, cls, popitself=True):
    # the first filter filters things that could not be classes
    # the second checks what we want
    L = filter(
        lambda x: issubclass(x, cls),
        filter(lambda x: type(x) == type(cls), L))
    rval = {l.__name__: l for l in L}

    if popitself:
        rval.pop(cls.__name__)
    return rval
