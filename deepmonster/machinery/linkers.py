import copy
from deepmonster.utils import (assert_iterable_return_iterable, issubclass_,
                               flatten, make_dict_cls_name_obj)
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
    def return_linkerclass(cls):
        linkers_cls_dict = make_dict_cls_name_obj(globals().values(), Linker, popitself=False)

        if isinstance(cls, str):
            try:
                rval = linkers_cls_dict[cls]
            except KeyError:
                raise KeyError("Invalid class name {} for a Linker class".format(cls))
            return rval
        elif cls in linkers_cls_dict.values():
            return cls
        else:
            raise ValueError("Does not recognize {} given to LinksHelper".format(cls))

    @staticmethod
    def filter_linkers(linkers, cls, filter_as_subclass=False):
        """Returns a list of linkers only of class / subclass cls
        """
        cls = LinksHelper.return_linkerclass(cls)

        if filter_as_subclass:
            filter_func = lambda x: issubclass_(x, cls)
        else:
            filter_func = lambda x: x.__class__ is cls
        return filter(filter_func, linkers)

    @staticmethod
    def broadcast_request(linkers, request):
        """Broadcast a request on the list of linkers triggering all their parse_request(request)
        and fetch their objects if they accept.
        """
        rval = []
        for link in linkers:
            _rval = link.parse_request(request)
            if _rval is not None:
                rval.append(_rval)

        return rval

    @staticmethod
    def filter_and_broadcast_request(linkers, request, cls):
        """Combine in a ~smart~ way filter_linkers and broadcast_request methods.
        ~smart~ means that this methods assemble the return result in typical usage
        of these calls depending on the class of linkers being filtered.

        CANNOT filter_as_subclass as this would break the ~smart~ return
        """
        cls = LinksHelper.return_linkerclass(cls)
        linkers = LinksHelper.filter_linkers(linkers, cls)
        rval = LinksHelper.broadcast_request(linkers, request)

        if cls in (TrackingLink, ModelParametersLink, AdjustSharedLink):
            return flatten(rval)
        elif cls is FunctionLink and len(rval) == 1:
            return rval[0]
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
                if link.__class__ == tmpl.__class__:
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
    a new instance of LinksHolder with these filtered links (except when
    calling methods involving 'broadcasts').
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
        if not issubclass_(other, LinksHolder):
            raise TypeError("operand + with LinksHolder only supported with another LinksHolder")
        return self.links + other.links

    @renew
    def filter_linkers(self, cls, filter_as_subclass=False):
        return LinksHelper.filter_linkers(self.links, cls, filter_as_subclass)

    def broadcast_request(self, request):
        return LinksHelper.broadcast_request(self.links, request, flatten_result)

    def filter_and_broadcast_request(self, request, cls):
        return LinksHelper.filter_and_broadcast_request(self.links, request, cls)

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


class LinkerObjError(Exception):
    pass


class Linker(object):
    """A linker is a type of object that carries another object. This other
    object can be of an arbitrary type. The goal of Linker is to define an
    interface to ease tossing around objects among all the pieces of the library.

    The insides of a linker can be structured in any way, but when called we want
    it to return an object. The class defines how linkers are compatible with
    others of the same type, if they hold what the caller wants and how to
    return the wanted object.
    """
    def __init__(self, obj):
        self.obj = obj
        self.assert_none()

    def assert_none(self):
        #TODO: make it possible to disable this assumption through DM library config
        """Defines how to check if an obj is treated as None and therefore rejected.
        There is no point in carrying around a None object and is actually a potential
        error causing problem. It would be extremely tedious to check everytime
        when fetching objects from linkers if they are None and we assume this is
        due to an error at their creation.
        """
        if self.obj is None:
            raise LinkerObjError("A None object is being linked")

    @property
    def raw_obj(self):
        """It is possible that the object being held is actually a Linker itself.
        raw_obj will go deep as possible to find the `real` obj
        """
        obj = self.obj
        while issubclass_(obj, Linker):
            try:
                obj = obj.obj
            except AttributeError:
                raise AttributeError("(THIS SHOULD NOT HAPPEN) Tried to find the raw object "+\
                                     "but stumbled upon a objectless link.")
        return obj


    def sanity_check(self, otherlink):
        """Raises an error if the test does not pass. Uses to define how
        a linker is compatible with another one. Ex.: Some linkers should have
        a unique name and do not tolerate another similarly named being created.

        By default does not raise anything.
        """
        return

    def parse_request(self, request):
        """Defines what to do when something calls for objects in the linker.
        This mechanism is optional and therefore is not mandatory to implement.
        """
        return NotImplementedError

    def accept_request(self, return_raw=True):
        """Returns the object when the request is accepted. This method is exposed
        if futher processing is required before returning object. However, a linker
        should never returns something else than self.obj else it diminishes the
        purpose of this interface.
        """
        if return_raw:
            return self.raw_obj
        return self.obj


class VariableLink(Linker):
    """Link for a model's variable.

    Obj is of type list
    """
    def __init__(self, var):
        var = assert_iterable_return_iterable(var, 'list')
        super(VariableLink, self).__init__(var)

    @property
    def var(self):
        return self.obj

    @property
    def raw_var(self):
        return self.raw_obj

    def assert_none(self):
        if any(x is None for x in self.var):
            raise LinkerObjError("Trying to link None variable(s)")


class GraphLink(VariableLink):
    """This linker annotates variables with a name and associate them with a graph.
    machinery.model.Model is based on this to be customizable and flexible.

    Obj is of type dict
    """
    def __init__(self, graph, var_name_dict=None):
        if var_name_dict is None:
            var_name_dict = {}
        self.graph = graph
        self.var_name_dict = var_name_dict
        super(VariableLink, self).__init__(var_name_dict)

    @property
    def var(self):
        return self.var_name_dict.values()

    @property
    def var_name(self):
        return self.var_name_dict.keys()

    def update(self, updt):
        self.var_name_dict.update(updt)

    def get_var_with_name(self, name):
        try:
            return self.var_name_dict[name]
        except KeyError:
            raise KeyError("Cannot find var named {} in graph {}".format(name, self.graph))

    def sanity_check(self, otherlink):
        if self.graph == otherlink.graph:
            raise ValueError("Two graphlinks cannot have the same name, "+\
                             "found a name collision with {}".format(self.graph))


class ParametersLink(VariableLink):
    """Indicates that this variable has these parameters / architectures linked to it.

    Obj is of type list
    """
    def __init__(self, var, parameters=None, architectures=None):
        super(ParametersLink, self).__init__(var)
        parameters = [] if parameters is None else parameters
        architectures = [] if architectures is None else architectures
        self.parameters = assert_iterable_return_iterable(parameters, 'list')
        self.architectures = assert_iterable_return_iterable(architectures, 'list')


###-------------------###
### EXTENSION LINKERS ###
###-------------------###
class ExtensionLink(Linker):
    """Indicates that this linker is useful for main loop extensions.
    When the training script builds the extensions, it will try to look for
    linkers that are associated with each extension. This is done
    at the ExtensionsFactory script's level.
    """
    def parse_request(self, request=None):
        return self.accept_request()


class TrackingLink(VariableLink, ExtensionLink):
    """Track a var and link it to a set. Typically for monitoring.

    Obj is of type list
    """
    def __init__(self, var, which_set='all'):
        super(TrackingLink, self).__init__(var)
        self.which_set = assert_iterable_return_iterable(which_set)

    def parse_request(self, request):
        if 'all' in self.which_set or request in self.which_set:
            return self.accept_request(return_raw=True)


class AdjustSharedLink(ExtensionLink):
    """Adjust a shared variable with some rules defined in the obj

    Obj is type obj with method 'adjust' that is implemented
    """
    def __init__(self, obj):
        assert hasattr(obj, 'adjust'), \
                "Incorrect object given to AdjustSharedLink. The method 'adjust' needs to exists."
        super(AdjustSharedLink, self).__init__(obj)


class UniquelyNamedLink(ExtensionLink):
    """ExtensionLinks that should be uniquely named
    """
    def __init__(self, obj, name=''):
        super(UniquelyNamedLink, self).__init__(obj)
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

    def parse_request(self, request=None):
        """These type of links can have None as request. This is interpreted
        as the fetcher expecting no other link of this type has been created
        and fetching by name is useless. The check of uniqueness should
        have been done beforehands.
        """
        if request is None or self.name == request:
            return self.accept_request()


class FunctionLink(UniquelyNamedLink):
    """Hold variables for a function to be built in the extensions.

    Obj is of type tuple
    """
    def __init__(self, inputs=[], outputs=[], **kwargs):
        inputs = assert_iterable_return_iterable(inputs)
        outputs = assert_iterable_return_iterable(outputs)
        obj = (inputs, outputs)
        super(FunctionLink, self).__init__(obj, **kwargs)


class StreamLink(UniquelyNamedLink):
    """Pass on the streams to the extensions.

    Obj is of type fuel.stream
    """
    def __init__(self, stream, **kwargs):
        self.stream = stream
        super(StreamLink, self).__init__(stream, **kwargs)


class ModelParametersLink(UniquelyNamedLink):
    """Pass on the parameters to the extensions.

    Obj is of type list
    """
    def __init__(self, parameters, **kwargs):
        self.parameters = assert_iterable_return_iterable(parameters)
        super(ModelParametersLink, self).__init__(parameters, **kwargs)
