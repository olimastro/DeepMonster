"""
    This script is an attempt to stop copy pasting
    everytime new scripts for every new experiments

    This script should be the template, in the sense
    that it should be the code base that never changes

    The moving pieces of an experiment are:
     - hard config values (batch size, ...)
     - graph from input to cost
     - model's architecture for each step in the graph
     - type of algorithm for bprop
     - mainloop extensions
     - fuel streams

    To not get lost in all the different combinations,
    maybe have one file containing all these spec
    and this script only parse them?
"""
import imp, yaml
from abc import ABCMeta, abstractmethod, abstractproperty
from inspect import isclass

from machinery import onlynewkeysdict
from machinery.runner import TrainModel
from machinery.datafetcher import DeepmonsterFetcher
from utils import merge_dicts, issubclass_, assert_iterable_return_iterable

# the scripts here are made to be more tight in their error checking
# we want the program to crash when the core is being setup if it finds
# an error as it would be really painfull for errors to silently
# slip by when an experiment is running
def hatch(path_to_yml, AssembleClass=Assemble):
    """Hatch a deep learning monster! Function to launch an experiment, it only
    needs one yml pointing to every pieces it needs to put together.

    Format of the masteryml:
        -----------------------------------------------
        model*: path/to/model.py
        architecture*: path/to/architecture.py
        config**: path/to/config.yml

        (OPTIONAL)
        runner*: path/to/runner.py
        (Will source TrainModel as a runner by default)

        dataset:
            name: name_of_dataset
            (optional)
            path: path/to/datafetcher.py
            option1: value1
            ...
        (Will source EmptyDataFetcher by default if not given)
        (If path to a fetcher is not given, it will source based on 'name'
         a default dataset defined in deepmonster.fuel.streams)
        -----------------------------------------------

        *2nd accepted format:
            path: path/to/architecture.py
            name: name_of_class_to_import

        (EVERY config.yml have to abide by these two possible formats)
        **2nd accepted format:
            path:
                - path/to/config1.yml
                  path/to/config2.yml
                  ...
            option1: value1
            option2: value2
            ...

         ***2nd accepted format:

    hatch requires two things two run: config values and an assemble object.
    The assemble object can be the library's default. This one also defines
    what are the essential python objects required to hatch a monster.
    The config values need to be in masteryml['config']
    """
    # loading config
    with open(path_to_yml, 'r') as f:
        masteryml = yaml.load(f)

    try:
        yml_config_vals = masteryml['config']
    except KeyError:
        raise KeyError("No configuration values found! Is there really an \
                       experiment that have zero config values?")
    config = parse_config(yml_config_vals)

    # load classes
    assemble = AssembleClass()
    core_classes = {s: None for s in assemble.core_pieces}

    def get_class(request):
        try:
            yml_val = masteryml[request]
        except KeyError:
            raise KeyError("{} field missing from yml file. Cannot ".format(req_core) + \
                           "start library without it")

        format_parser = assemble.format_cls_dict[request](request)
        format_parser.check(yml_val)
        return format_parser.load_with_this_format(yml_val)

    for req_core in assemble.cores_mandatory_to_init:
        core_classes[req_core] = get_class(req_core)

    for opt_core in assemble.cores_with_default:
        try:
            get_class(req_core)
        except KeyError:
            core_classes[req_core] = assemble.core_pieces_init_dict[req_core]

    # init classes
    core_instances = {s: None for s in assemble.core_pieces}
    for cls in assemble.core_pieces:
        if len(parser_dict[cls].particular_extra_config) > 0:
            particular_config = merge_dicts(config, parser_dict[cls].particular_extra_config)
        else:
            particular_config = config

        instance = core_classes[cls](particular_config)

    for instance in core_instances.values():
        for cd in instance.__core_dependancies__:
            instance.bind_a_core(cd, core_instances[cd])

    # hatch the monster!
    core_instance[assemble.runnable].run()


class Assemble(object):
    """An assemble object of the deep monster library defines the core pieces that
    are entrypoints when launching the main script. They are child of the Core class
    and typically defined in other python script.

    Assemble's job is to provide a setup of core objects that make the library do
    its purpose. It can actually be anything, and the entry script, hatch, will
    query Assemble for all its Core classes and finish by calling 'run()' on what
    Assemble tells it is runnable.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        self.core_pieces = self.core_pieces_init_dict.keys()
        self.cores_with_default = [x for x,v in self.core_pieces_init_dict.iteritems() if v is not None]
        self.cores_mandatory_to_init = list(set(core_pieces) - set(self.cores_with_default))

    @abstractproperty
    def core_pieces_init_dict(self):
        cpid = {
            'architecture': None,
            'model': None,
            'runner': TrainModel,
            'dataset': DefaultFetcher,
        }
        return cpid

    @abstractproperty
    def format_cls_dict(self):
        bf = {
            'architecture': StandardFormat,
            'model': StandardFormat,
            'runner': StandardFormat,
            'dataset': DefaultNamedFormat,
        }
        return bf

    @abstractproperty
    def runnable_core(self):
        return 'runner'


class YmlParser(object):
    """Parser base class that parses yml files. Every possible format a yml field
    can take is defined into the hierachy of all its class child. A YmlParser
    child also has to define the error message if the format it enforces is not
    respected and on the opposite, if the format is found to be correct, how
    to load the object.

    This is done in a hierachical fashion among the child classes
    iteratively checking the format until a correct one is found. The order of
    checks can be overriden by specifying the field __hierarchy__. A child can
    also override an error message of a parent by decorating a method with
    YmlParser.override_error_msg(parent_class).

    For example, the format that the field 'model' can take is StandardFormat.
    StandardFormat is a subclass of SimpleFormat. When parsing 'model' in the yml file,
    YmlParser will go hierachically in reverse __mro__ (SimpleFormat, StandardFormat)
    and call isformat method of each child class. When the correct format is found, it
    will bind the load_with_this_format method to this specific Format.
    """
    __metaclass__ = ABCMeta
    __override_msg__ = {}

    def __init__(self, key):
        self.key = key
        self.particular_extra_config = {}
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

    def load_with_this_format(self, x):
        """Specify how to load the key when this format is found.
        """
        raise NotImplementedError("Base YmlParser class does not know how to load a yml key")

    def load(self, x):
        return self.__mro__[self.format].load_with_this_format(x)

    @classmethod
    def override_error_msg(cls, target_cls):]
        def decorator(func):
            cls.__override_msg__.update({target_cls:func})
            return func
        return decorator

    def get_order(self):
        if '__hierarchy__' in self.__dict__:
            order = self.__hierarchy__
        else:
            order = self.__mro__[:-2].reverse()
        return order

    def check(self, value):
        for cls in self.get_order():
            isformat = cls.isformat(value)
            if isformat:
                self.format = cls
                return
        if self.format is None:
            raise Exception(self.on_error(value))

    def on_error(self, value):
        msg = "Encountered error into parsing {}, {}".format(self.key, value)
        msg += '\n'
        msg = "In order to run, {} can have these following formats:\n".format(self.key)
        for cls in self.get_order():
            msg += '\n'
            if cls in self.__override_msg__.keys():
                msg += self.__override_msg__[cls]() + '\n'
            else:
                msg += cls.error_msg() + '\n'
        return msg


class SimpleFormat(YmlParser):
    """field: path/to/field.[filetype]
    """
    def __init__(self, key, filetype='.py'):
        if '.' != filetype[0]:
            filetype = '.' + filetype
        self.filetype = filetype
        super(SimpleFormat, self).__init__(key)

    def isformat(self, x):
        if self.is_file_type(x):
            return True
        return False

    def error_msg(self):
        msg = "path/to/{}.py".format(self.key)
        return msg

    def is_file_type(self, x):
        if isinstance(x, str):
            if x.split('/')[-1].split(self.filetype)[-1] == '':
                return True
        return False

    def load_with_this_format(self, x):
        return import_class_from_path(x)


class StandardFormat(SimpleFormat):
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
        msg = "path: path/to/{}.py \n".format(self.key) + \
                "name: name_of_class_to_import in {}".format(self.filetype)
        return msg

    def load_with_this_format(self, x):
        return import_class_from_path(x['path'], name=x['name'])


class RecursivelyDefinedFormat(SimpleFormat):
    """Special format for config file. Its parent
    is SimpleFormat so config can be as 'config: path/to/config.yml'.
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


class DefaultNamedFormat(StandardFormat):
    """name: standard name that the library knows how to handle.
    This format also tolerates more config values to pass on
    to its object. These values won't appear to the global
    config dict and are particular for this object.
    """
    __hierarchy__ = [DefaultNamedFormat, StandardFormat]

    def __init__(self, key, default_names):
        self.default_names = default_names
        super(DefaultNamedFormat, self).__init__(key, '.py')

    def isformat(self, x):
        if isinstance(x, str):
            if x in self.default_names:
                return True

    def error_msg(self):
        msg = 'name: name_of_default_dataset_to_fetch\n' + \
                'Availables default_dataset: {}'.format(
                    DeepmonsterFetcher.__knows_how_to_load__)
        return msg

    @YmlParser.override_error_msg('StandardFormat')
    def sdf_error_msg(self):
        msg = "path: path/to/{}.py \n".format(self.key) + \
                "name: name_of_dataset\n\
                options1: value1\n\
                options2: value2\n\
                ..."
        return msg

    def load_with_this_format(self, x):
        self.particular_extra_config.update({self.key:x})
        super(DefaultNamedFormat, self).load_with_this_format(x)


def import_class_from_path(pypath, name=None):
    """Get the class requested in pypath.
    """
    # lets hope that binding everything to 'core_imports' is fine?
    mod = imp.load_source('core_imports', pypath)
    # when no name is given, we can only guess what the user want
    if name is None:
        nb_imported_cls = sum(map(isclass, map(lambda x: getattr(mod, x), dir(mod))))
        if nb_imported_cls < 1:
            raise ImportError("There are no class to import in {}".format(pypath))
        elif nb_imported_cls > 1:
            print "WARNING: More than one class is imported in {} and name is \
                    undefined, could be picking the wrong one".format(pypath)
        # hope for the best
        for mod_name in dir(mod):
            cls = getattr(mod, mod_name)
            if issubclass_(cls, Configurable):
                return cls
    else:
        return getattr(mod, name)


def parse_config(configyml):
    """Parse the configuration file. The argument here is either a string
    pointing to a yml, or a dict with config values and potentially
    a 'path' key pointing to other yml to join. It will raise an error
    if two keys have the same name (except for path).

    Returns a merge of all the config files as the final configuration dictionnary.
    """
    format_parser = RecursivelyDefinedFormat('config', '.yml')
    def update_dict(yml, D):
        _yml = yml.copy().pop('path')
        D.update(_yml)

    def recursive_parse(yml, D):
        format_parser.check(yml)
        if isinstance(yml, dict):
            if yml.has_key('path'):
                for p in assert_iterable_return_iterable(yml['path']):
                    recursive_parse(p, D)
            update_dict(yml, D)
            return

        #assert isinstance(yml, str) and '.yml' in yml, "Problem occured in recursive parsing"
        with open(yml, 'r') as f:
            next_yml = yaml.load(f)
        recursive_parse(next_yml, D)

    D = onlynewkeysdict({})
    recursive_parse(configyml, D)
    return D
