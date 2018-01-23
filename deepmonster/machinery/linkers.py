import copy
from deepmonster.utils import (assert_iterable_return_iterable, issubclass_,
                               make_dict_cls_name_obj)
"""
    Along with Core, Links are the fundamental design pieces of deepmonster.
    While Core define the interface for dictionary configurable objects, Links
    define an interface for all objects that are passed around throughout the
    library's execution.

    When a Core, or another piece of deepmonster, wants something of another
    Core, these actions and requests should be done through the standarized
    way that Linkers try to achieve. As anything in python, this is not strictly
    enforced and there are always other ways to hack around. At your own risk :)
"""

# The logic that follows with Helper and Holder is that Helper should define
# stateless method that can do things on a _list_. Holder meanwhile subclasses
# the same methods but act on its own list of linkers and return a new instance
# of Holder.

class LinksHelper(object):
    """Helper class to interface between Linkers and other scripts.
    """
    @staticmethod
    def filter_linkers(linkers, cls):
        if isinstance(cls, str):
            linkers_cls_dict = make_dict_cls_name_obj(globals().values(), Linker)
            cls = linkers_cls_dict[cls]
        return filter(lambda x: x.__class__ is cls, linkers)

    @staticmethod
    def broadcast_request(linkers, request):
        rval = []
        for link in linkers:
            _rval = link.parse_request(request)
            if _rval is not None:
                rval.append(_rval)
        return rval

    @staticmethod
    def sanity_check(linkers):
        assert all(issubclass_(l, Linker) for l in linkers), "Non Linker(s) detected"
        if len(linkers) < 2:
            return
        tmpcp = copy.copy(linkers)

        for link in linkers:
            # do not need to check this link against itself
            tmpcp.remove(link)
            for tmpl in tmpcp:
                if link.__class__ == tmpl.__class:
                    link.sanity_check(tmpl)
        del tmpcp

    @staticmethod
    def merge_holders(holders):
        rval = LinksHolder()
        for holder in holders:
            rval += holder
        return rval


class LinksHolder(LinksHelper):
    """Knows every link that has been instanciated during a model's
    definition of its graph.

    This class inherits from Helper so its util methods can be called
    on a LinksHolder instance. It will apply them and return the relevant
    values. The variation is that if filter is called, it will return
    a new instance of LinksHolder with these filtered links.
    """
    def __init__(self, links=None):
        links = [] if links is None else \
                assert_iterable_return_iterable(links, 'list')
        LinksHelper.sanity_check(links)
        self._links = links

    def __iter__(self):
        def linksholder_iterator():
            for link in self.links:
                yield link
        return linksholder_iterator()

    def __len__(self):
        return len(self.links)

    def renew(func):
        """Method decorated with renew simply needs
        to return a list of links and renew will take care to
        make a new instance.
        """
        def _renew(*args, **kwargs):
            l = func(*args, **kwargs)
            return LinksHolder(l)
        return _renew

    @renew
    def __add__(self, other):
        if other is not LinksHolder:
            raise TypeError("operand + with LinksHolder only supported with another LinksHolder")
        return self.links + other.links

    @renew
    def filter_linkers(self, cls):
        return LinksHelper.filter_linkers(self.links, cls)

    @property
    def links(self):
        return self._links

    def store_links(self, newlinks):
        newlinks = assert_iterable_return_iterable(newlinks, 'list')
        # execute checks for all new links among themselves
        LinksHelper.sanity_check(newlinks)

        # execute checks for all new links compared to existing ones
        for newlink in newlinks:
            for link in self.links:
                # can we compare a link with a link of a diff class?
                if link.__class__ == newlink.__class__:
                    link.sanity_check(newlink)

        self._links.extend(newlinks)

    def clear_links(self):
        print "WARNING: Links are being cleared, hope this was intended"
        self._links = []



class Linker(object):
    """
    Helpers for the many different usage a variables
    in the graph can have in the model and / or with training to pass them
    around to the objects that need them. A typical
    use of these links would for example if the variables are needed
    for an extension or are a special cost in the bprop graph.
    """
    # unused in current implementation but lets keep it
    @classmethod
    def check_condition(cls, field, condition):
        """A linker can have specifics fields and conditions to be checked
        wheter that's the linker we are looking for. Each field
        to be checked should have a classmethod associated to it
        so we can check if the condition regarding this field is met.

        By default, will return false. This has the drawback to silence a
        potential SyntaxError where the user was asking for 'extnsion' but
        the real field is 'extension'.
        """
        try:
            return getattr(cls, 'check_'+field)(condition)
        except AttributeError:
            return False

    def sanity_check(self, otherlink):
        # this should raise an error if the test does not pass
        return


class VariableLink(Linker):
    """Link for a model's variable.
    """
    def __init__(self, var):
        # it would make no sense to have the same var more than once?
        self.var = assert_iterable_return_iterable(var, 'set')

    @property
    def model_var(self):
        # some var could be deep in a multiple Link hiearchy, this
        # returns the raw model variable
        var = self.var
        while issubclass(var, VariableLink):
            try:
                var = var.var
            except AttributeError:
                raise AttributeError("Tried to find raw model variables "+\
                                     "but stumbled upon a variableless link")
        return var


class GraphLink(VariableLink):
    """
        This linker annotates variables with a name and
        associate them with a graph. Model is based on
        this to be customizable and ease the tossing
        around of variables.
    """
    def __init__(self, graph, var_name_dict={}):
        self.graph = graph
        self.var_name_dict = var_name_dict

    @property
    def var(self):
        return set(self.var_name_dict.keys())

    def update(self, updt):
        self.var_name_dict.update(updt)

    def get_var_wth_name(self, name):
        for key, value in self.var_name_dict.iteritems():
            if value == name:
                return key
        raise KeyError("Cannot find var named {} in graph {}".format(name, self.graph))

    def sanity_check(self, otherlink):
        if self.graph == otherlink.graph:
            raise ValueError("Two graphlinks cannot have the same name, "+\
                             "found a name collision with {}".format(self.graph))


class ParametersLink(VariableLink):
    """Indicates that this variable has these parameters linked to it
    """
    def __init__(self, var, parameters=[]):
        super(ParametersLink, self).__init__(var)
        self.parameters = assert_iterable_return_iterable(parameters)


###-------------------###
### EXTENSION LINKERS ###
###-------------------###
class ExtensionLink(Linker):
    """Indicates that this linker is useful for main loop extensions.
    When the training script builds the extensions, it will try to look for
    linkers that are associated with each extension. This is done
    at the ExtensionsFactory script's level.

    It expects these linkers to have the `parse_request` method implemented.
    """
    def parse_request(self, request):
        raise NotImplementedError()

class TrackingLink(VariableLink, ExtensionLink):
    """Track a var and link it to a set. Typically for monitoring.
    """
    def __init__(self, var, which_set=(None,)):
        super(TrackingLink, self).__init__(var)
        self.which_set = assert_iterable_return_iterable(which_set)

    def parse_request(self, request):
        if 'all' in self.which_set or request in self.which_set:
            return self.var


class UniquelyNamedLink(ExtensionLink):
    """ExtensionLinks that should be uniquely named
    """
    def __init__(self, name=''):
        self.name = name

    def sanity_check(self, otherlink):
        if self.name == otherlink.name:
            if self.name == '':
                raise RuntimeError(
                    "Trying to create two links of type {} without giving them names. ".format(
                        self.__class__)+\
                    "These types of links are also UniquelyNamedLink and only one link of this "+\
                    "type is tolerated without a name. With more than one unanmed, they "+\
                    "become undstinguishiable and the run might become unstable.")
            raise ValueError(
                "Collision found for same name {} in ExtensionLinkers".format(self.name))

    def parse_request(self, request):
        # These special links can have None as request. This is interpreted
        # as the fetcher expecting no other link of this type has been created
        # and fetching by name is useless. The check of uniqueness should
        # have been done beforehands.
        if request is None or self.name == request:
            return self.accept_request()


class FunctionLink(UniquelyNamedLink):
    """Hold variables for a function to be built in the extensions.
    """
    def __init__(self, inputs=[], outputs=[], **kwargs):
        super(FunctionLink, self).__init__(**kwargs)
        self.inputs = assert_iterable_return_iterable(inputs)
        self.outputs = assert_iterable_return_iterable(outputs)

    def accept_request(self):
        return (self.inputs, self.outputs,)


class StreamLink(UniquelyNamedLink):
    """Pass on the streams to the extensions.
    """
    def __init__(self, stream, **kwargs):
        super(StreamLink, self).__init__(**kwargs)
        self.stream = stream

    def accept_request(self):
        return self.stream


class ModelParametersLink(UniquelyNamedLink):
    """Pass on the parameters to the extensions.
    """
    def __init__(self, parameters, **kwargs):
        super(ModelParametersLink, self).__init__(**kwargs)
        self.parameters = assert_iterable_return_iterable(parameters)

    def accept_request(self):
        return self.parameters
