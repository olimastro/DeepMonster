import imp, yaml, os, sys, re
from inspect import isclass
from string import Template

from deepmonster.utils import issubclass_, assert_iterable_return_iterable
from core import Core
from dicttypes import onlynewkeysdict

# parse tuple in the yaml as python tuple not strings
# but the !!python/tuple tag wants us to write "tuple: [230,123]", we want the tuple syntax
def yml_tuple_constructor(loader, node):
    def parse_tup_el(el):
        # try to convert into int or float else keep the string
        if el.isdigit():
            return int(el)
        try:
            return float(el)
        except ValueError:
            return el

    value = loader.construct_scalar(node)
    # remove the ( ) from the string
    tup_elements = value[1:-1].split(',')
    # remove the last element if the tuple was written as (x,b,)
    if tup_elements[-1] == '':
        tup_elements.pop(-1)
    tup = tuple(map(parse_tup_el, tup_elements))
    return tup

yaml.add_constructor(u'!tuple', yml_tuple_constructor)
yaml.add_implicit_resolver(u'!tuple', re.compile(r"\(([^,\W]{,},){,}[^,\W]*\)"))


class KeyFormatError(Exception):
    def __init__(self, key, err_msg='KeyError: $key'):
        self.key = key
        if not hasattr(self, 'err_msg'):
            self.err_msg = Template(err_msg)

    def __str__(self):
        return repr(self.err_msg.substitute(key=self.key))

class PythonFormatError(KeyFormatError):
    err_msg = Template("""
When specifying a python script, only an entry of types:\n\
$key: {path: path/to/python, name: name of class to load}".format(self.key_for_python)\n\
OR\n\
$key: path/to/python""")

class YmlFormatError(KeyFormatError):
    err_msg = Template("""
Wrong format for $key while trying to look for paths pointing to other yml files.\
Should be str or list""")


class YmlFieldNotFoundError(IOError):
    pass


class YmlParser(object):
    """Directly handles paths to yml files and parse entries.
    """
    def __init__(self, path_to_yml, mandatory_fields=None, preprocess=True,
                 key_for_ymlpaths='ymlpath', key_for_python='pypath'):

        self.path_to_yml = path_to_yml
        self.mandatory_fields = [] if mandatory_fields is None else \
                assert_iterable_return_iterable(mandatory_fields)
        self.preprocessing = preprocess
        self.key_for_ymlpaths = key_for_ymlpaths
        self.key_for_python = key_for_python


    def preprocess(self, pydict):
        """The only current preprocessing available on yml files is to replace all
        bash type references such as $HOME with their values
        """
        def parse(s):
            return os.path.normpath(os.path.expandvars(s))

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

        if not all(x in yml.keys() for x in self.mandatory_fields):
            # find missing key
            missing = set(self.mandatory_fields) - set(yml.keys)
            raise YmlFieldNotFoundError("{} field(s) missing from yml file. Cannot "+\
                                        "start library without it".format(missing))

        if self.preprocessing:
            self.preprocess(yml)
        self.yml = yml


    def load_ymlentry(self, ymlkey):
        """Load a yml entry from the yml dict.

        Returns the full dict from this entry and a class (potentially None)
        """
        # this could still happen for optional fields
        try:
            yml_val = self.yml[ymlkey]
        except KeyError:
            raise YmlFieldNotFoundError(ymlkey + " not found")

        if not isinstance(yml_val, dict):
            err_msg = "When loading entry $key it should return a dict "+\
                    "and it is not. Likely an error due to the format used in the yml"
            raise KeyFormatError(yml_key, err_msg)

        class_to_load = None

        if yml_val.has_key(self.key_for_python):
            python_info = yml_val[self.key_for_python]
            name = None
            if isinstance(python_info, str):
                if not os.path.isfile(python_info):
                    raise IOError("Cannot find {}".format(python_info))
                pypath = python_info

            elif isinstance(python_info, dict):
                if not (python_info.has_key('path') and python_info.has_key('name')):
                    raise PythonFormatError(self.key_for_python)

                pypath = python_info['path']
                name = python_info['name']

            else:
                raise PythonFormatError(self.key_for_python)

            class_to_load = import_class_from_path(pypath, name)

        yml_rval = unpack_yml(yml_val, self.key_for_ymlpaths)

        return yml_rval, class_to_load


    @classmethod
    def load_from_path(cls, path_to_yml, mandatory_fields=None, preprocess=True):
        """Bypass initialization and instantly return the loaded yml file
        """
        loader = cls(path_to_yml, mandatory_fields, preprocess)
        loader.load()
        return loader.yml



def unpack_yml(configyml, key_for_ymlpaths='ymlpath'):
    """Unpack a ymlpath to a pydict and merge it the current dict.
    Returns the fully merged dict.
    """
    def update_dict(yml, D):
        _yml = yml.copy()
        _yml.pop(key_for_ymlpaths, None)
        D.update(_yml)

    def recursive_parse(yml, D):
        if isinstance(yml, dict):
            if yml.has_key(key_for_ymlpaths):
                if not isinstance(yml[key_for_ymlpaths], (str, list)):
                    raise YmlFormatError(key_for_ymlpaths)

                for p in assert_iterable_return_iterable(yml[key_for_ymlpaths]):
                    if not os.path.isfile(p):

                        raise IOError("Cannot find {}".format(p))
                    recursive_parse(p, D)

            update_dict(yml, D)
            return

        next_yml = YmlLoader.load_from_path(yml)
        recursive_parse(next_yml, D)

    D = onlynewkeysdict({})
    recursive_parse(configyml, D)
    return D


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
