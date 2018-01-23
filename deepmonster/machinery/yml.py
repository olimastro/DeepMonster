import imp, yaml, os, sys
from inspect import isclass
from abc import ABCMeta, abstractmethod, abstractproperty

from deepmonster.utils import issubclass_, assert_iterable_return_iterable
from core import Core

#TODO: Found a bug in the design of this whole format buisness. For now,
# make it workable. The TODO for future versions imply having even more
# seperations in the class. One for the dict format the pairs (key, value)
# and another one for one given pair aka on said key what's the correct value

class FormatError(Exception):
    pass

class YmlFieldNotFoundError(Exception):
    pass


class YmlLoader(object):
    """Directly handles paths to yml files and coordinates loading depending
    of what format YmlParser has found is matching.
    """
    def __init__(self, path_to_yml, mandatory_fields=None, preprocess=True):
        self.path_to_yml = path_to_yml
        self.mandatory_fields = [] if mandatory_fields is None else \
                assert_iterable_return_iterable(mandatory_fields)
        self.preprocessing = preprocess

    def preprocess(self, pydict):
        """The only current preprocessing available on yml files is to replace all
        bash type references such as $HOME with their values
        """
        def parse(s):
            new_s = s
            for env in os.environ:
                if '$'+env in s:
                    new_s = s.replace('$'+env, os.environ[env])
                    break
            return new_s

        for k,v in pydict.iteritems():
            new_val = None
            if isinstance(v, list):
                new_val = [parse(i) for i in v]
            elif isinstance(v, dict):
                self.preprocess(v)
            elif isinstance(v, str):
                new_val = parse(v)
            else:
                continue
            if new_val is not None:
                pydict[k] = new_val

    def load(self):
        """Load the yml dict from the path and potentially do some preprocessing.
        Set to self.yml the dict.
        """
        with open(self.path_to_yml, 'r') as f:
            yml = yaml.load(f)

        try:
            assert all(x in yml.keys() for x in self.mandatory_fields)
        except AssertionError:
            # find missing key
            missing = set(self.mandatory_fields) - set(yml.keys)
            raise AssertionError("{} field(s) missing from yml file. Cannot \
                                 start library without it".format(missing))

        if self.preprocessing:
            self.preprocess(yml)
        self.yml = yml

    def load_ymlentry(self, ymlkey, format_parser):
        """Load a yml entry from the yml dict according to its parsing. If an unrecognized
        format of the dict was given, format_parser will throw an error and abort.
        """
        # this could still happen for optional fields
        try:
            yml_val = self.yml[ymlkey]
        except KeyError:
            raise YmlFieldNotFoundError(ymlkey + " not found")

        format_parser.check(yml_val)
        return format_parser.load(yml_val)

    @classmethod
    def load_from_path(cls, path_to_yml, mandatory_fields=None, preprocess=True):
        """Bypass initialization and instantly return the loaded yml file
        """
        loader = cls(path_to_yml, mandatory_fields, preprocess)
        loader.load()
        return loader.yml


class YmlParser(object):
    """Parser base class that parses yml files that are read through YmlLoader.
    Basically, this class parses a *python dictionary* (since that is the result
    of yaml.load(file.yml)) and check if its structure is correct.

    The definition of a 'correct dictionary structure' is done through the
    hierachy of all its class children. Each child also has to define the
    error message if the format it enforces is not respected
    On the opposite, if the format is found to be correct, it defines
    what is the next step to do with the dictionary (most likely defers to
    a function that loads a python class)

    This is done in a hierachical fashion among the child classes
    iteratively checking the format until a correct one is found. The order of
    checks can be overriden by specifying the field __hierarchy__. A child can
    also override an error message of a parent by decorating a method with
    YmlParser.override_error_msg(parent_class).

    For example:
        The format that the field 'model' can take is StandardFormat. StandardFormat
        is a subclass of FilePathFormat. When parsing 'model' in the yml file,
        YmlParser will go hierachically in reverse __mro__ (FilePathFormat, StandardFormat)
        and call 'isformat' method of each child class.

        Let's say model.yml was defined:
            model:
                path: $HOME/model.py
                name: FooModel

        FilePathFormat.isformat() returns False, but StandardFormat.isformat() returns
        True. We now know that we need to take 'load_with_this_format' of StandardFormat
        for further processing.
    """
    __metaclass__ = ABCMeta
    __override_msg__ = {}

    def __init__(self, key):
        self.key = key
        self.format = None

    @abstractmethod
    def isformat(self, x):
        """Return True / False if x is in the correct format. These are built in
        a hierachical fasion and YmlParser will call *every* isformat of the __mro__.
        """
        pass

    @abstractmethod
    def error_msg(self):
        """Return the error message to the user. These are built in a hierachical
        fashion and YmlParser will aggregate all the error messages in the __mro__
        """
        pass

    @property
    def class_tree_list(self):
        return list(self.__class__.__mro__)

    def load_with_this_format(self, x):
        """Specify how to load the key when this format is found.
        """
        raise NotImplementedError("Base YmlParser class does not know how to load a yml key")

    def load(self, x):
        i = self.class_tree_list.index(self.format)
        Format = self.class_tree_list[i]
        return Format.load_with_this_format(self, x)

    @classmethod
    def override_error_msg(cls, target_cls):
        def decorator(func):
            cls.__override_msg__.update({target_cls:func})
            return func
        return decorator

    def get_order(self):
        if '__hierarchy__' in self.__dict__:
            order = self.__hierarchy__
            if 'self' in order:
                order[order.index('self')] = self.__class__
        else:
            order = self.class_tree_list[:-2]
            order.reverse()
        return order

    def check(self, value):
        for cls in self.get_order():
            isformat = cls.isformat(self, value)
            if isformat:
                self.format = cls
                return
        raise FormatError(self.on_error(value))

    def on_error(self, value):
        msg = "\nEncountered error into parsing {}, {}".format(self.key, value)
        msg += '\n'
        msg += "In order to run, {} can have these following formats:\n".format(self.key)
        for cls in self.get_order():
            msg += '\n'
            if cls in self.__override_msg__.keys():
                msg += self.__override_msg__[cls]() + '\n'
            else:
                msg += cls.error_msg(self) + '\n'
        return msg


class FilePathFormat(YmlParser):
    """field: path/to/field.[filetype]
    """
    def __init__(self, key, filetype='.py'):
        if '.' != filetype[0]:
            filetype = '.' + filetype
        self.filetype = filetype
        super(FilePathFormat, self).__init__(key)

    def isformat(self, x):
        if not isinstance(x, str):
            return False
        if not os.path.isfile(os.path.realpath(x)):
            return False
        if self.is_file_type(x):
            return True
        return False

    def error_msg(self):
        msg = "path/to/{}{}".format(self.key, self.filetype)
        return msg

    def is_file_type(self, x):
        if isinstance(x, str):
            if x.split('/')[-1].split(self.filetype)[-1] == '':
                return True
        return False

    def load_with_this_format(self, x):
        return import_class_from_path(x)


class StandardFormat(FilePathFormat):
    """field:
        path: path/to/field.[filetype]
        name: name of what's looked for
    """
    def isformat(self, x):
        if not isinstance(x, dict):
            return False
        if len(x) == 2:
            if x.has_key('name') and x.has_key('path'):
                if self.is_file_type(x['path']):
                    return True
        return False

    def error_msg(self):
        msg = "path: path/to/{}{} \n".format(self.key, self.filetype) + \
                "name: name of class to import in {}".format(self.filetype)
        return msg

    def load_with_this_format(self, x):
        return import_class_from_path(x['path'], name=x['name'])


class RecursivelyDefinedFormat(FilePathFormat):
    """Special format for config file. Its parent
    is FilePathFormat so config can be as 'config: path/to/config.yml'.
    It can also contain other fields with any number of paths
    pointing to more yml files which are to be joined all together.

    All the other fields are config values.
    """
    def isformat(self, x):
        if not isinstance(x, dict):
            return False
        if x.has_key('path'):
            try:
                assert isinstance(x['path'], list)
                assert(all(self.is_file_type(y) for y in x['path']))
            except AssertionError:
                return False
        return True

    def error_msg(self):
        msg = "path :\n -"
        return msg


#TODO: design bug found around the logic of this format. The requirements
# of this format prevent the usage of previous Format classes even though
# they are really close
class DefaultNamedFormat(FilePathFormat):
    """name: standard name that the library knows how to handle.
    This format also tolerates more config values to pass on
    to its object. These values won't appear to the global
    config dict and are particular for this object.
    """
    def __init__(self, key, default_names):
        self.default_names = default_names
        self._format_type = None
        super(DefaultNamedFormat, self).__init__(key)

    def isformat(self, x):
        if isinstance(x, str):
            if x in self.default_names:
                self._format_type = 1
                return True
        elif isinstance(x, dict):
            try:
                assert all(map(x.has_key, ['path','name']))
                assert FilePathFormat.isformat(self, x['path'])
            except AssertionError:
                return False
            self._format_type = 2
            return True
        return False

    def error_msg(self):
        msg = 'name: name of default {} to fetch'.format(self.key)
        msg += '\n OR\n'
        msg = "path: path/to/{}.py \n".format(self.key) + \
                "(**the .py should have stricly ONE Core class**)\n\
                name: name value\n\
                options1: value1\n\
                options2: value2\n\
                ..."
        return msg

    def load_with_this_format(self, x):
        if self._format_type == 1:
            return None
        return super(DefaultNamedFormat, self).load_with_this_format(x['path'])


def import_class_from_path(pypath, name=None, class_type=Core):
    """Get the class requested in pypath.
    When name is None, it will search the classes being defined in the pypath
    for a subclass of class_type.
    """
    assert name is not None or class_type is not None
    ### ---------------------------------------------------------------------- ###
    #NOTE: going by imp just doesnt work right, you need to specify a name
    # argument to bind the imported modules and then it fu@ks up the namespace
    # of these modules even for things that were imported inside that file.
    # ex.:
    #   We call imp.load_source('core_imports.model, /some/path/model.py)
    #   and let's say there is a class Model defined in model.py which is a subclass
    #   of Core. Surprisingly, issubclass(Model, Core) == False!! The problem
    #   seems to be when checking Model.__mro__, you do see Core in the hierarchy,
    #   but no longer < deepmonster.machinery.core.Core >, it reads instead,
    #   < core_imports.model.Core > thus failing the subclass test :(
    ### ---------------------------------------------------------------------- ###

    # execute the path and collect the namespace
    namespace = {}
    execfile(pypath, namespace)
    class_obj = filter(lambda x: isclass(x), namespace.values())
    if len(class_obj) < 1:
        raise ImportError("There is no class to import in {}".format(pypath))

    if name is None:
        subclass_obj = filter(lambda x: issubclass_(x, class_type), class_obj)
        rval = filter(lambda x: x.__module__ == '__builtin__', subclass_obj)
        if len(rval) > 1:
            raise ImportError("Importation for a subclass of {} in {} ".format(class_type, pypath) +\
                              "failed. There are more than one subclass, the name is required")
    else:
        rval = filter(lambda x: x.__name__ == name, class_obj)
        if len(rval) > 1:
            raise ImportError("Importation for a class named {} in {} ".format(name, pypath) +\
                              "failed. More than one found")

    if len(rval) < 1:
        raise ImportError("There is no class to import in {} with name {} ".format(pypath, name) +\
                          "or subclass of type {}".format(class_type))
    return rval[0]
