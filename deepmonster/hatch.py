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

    To not get lost in all the different combinations,
    maybe have one file containing all these spec
    and this script only parse them?
"""
import imp, yaml
from inspect import isclass

from machinery import Configurable
from machinery.runner import TrainModel
from utils import issubclass_, assert_iterable_return_iterable

def hatch(path_to_yml):
    """Hatch a deep monster! Function to launch an experiment, it only
    needs one yml pointing to every pieces it needs to put together.

    Format of the masteryml:
        model*: path/to/model.py
        architecture*: path/to/architecture.py
        config**: path/to/config.yml

        (OPTIONNAL - else will source TrainModel by default)
        runner*: path/to/runner.py

        *2nd accepted format:
            path: path/to/architecture.py
            name: name_of_class_to_import

        (EVERY config.yml have to abide by these two possible formats)
        **2nd accepted format:
            path:
                - path/to/config1.yml
                - path/to/config2.yml
                ...
            option1: value1
            option2: value2
            ...

    hatch will gather the config dict from all over the possible given yml paths,
    instanciate Model and ModelRunner objects and finally, execute ModelRunner.run()
    """
    # finding paths
    with open(path_to_yml, 'r') as f:
        masteryml = yaml.load(f)

    try:
        path_arch = masteryml['architecture']
        path_model = masteryml['model']
        path_config = masteryml['config']
    except KeyError:
        raise KeyError("In order to run, the launch object requires \
                       at least three specification in the yml file: A path \
                       to the script defining the model, another path to \
                       the architecture, and a field defining config values")

    # optional config values
    path_runner = masteryml.get('runner', None)

    pypaths = [path_arch, path_model]
    if path_runner is not None:
        pypaths.append(path_runner)
    else:
        Runner = TrainModel

    assert all(map(lambda x: isinstance(x, str) or (x.has_key('path') and \
                                                    x.has_key('name') and \
                                                    len(x) == 2), pypaths)), \
            "The values of model and architecture (and optionnally runner if \
            specified) have to be either strings or two fields [path, name]"

    # parsing phase - parse config yml(s)
    config = parse_config(path_config)

    # parsing phase - import
    Classes = map(import_class_from_path, pypaths)

    # init the core pieces
    architecture = Classes[0](config)
    model = Classes[1](architecture, config)
    if path_runner is not None:
        runner = Classes[2](model, config)
    else:
        runner = TrainModel(model, config)

    # hatch the monster!
    runner.run()


def import_class_from_path(pypath):
    """Get the class requested in pypath.

    pypath can be a str or a dict with fields name and path.
    """
    name = None
    path = pypath
    if isinstance(pypath, dict):
        name = pypath['name']
        path = pypath['path']

    # lets hope that binding everything to 'core_imports' is fine?
    mod = imp.load_source('core_imports', path)
    # when no name is given, we can only guess what the user want
    if name is None:
        nb_imported_cls = sum(map(isclass, map(lambda x: getattr(mod, x), dir(mod))))
        if nb_imported_cls < 1:
            raise ImportError("There are no class to import in {}".format(path))
        elif nb_imported_cls > 1:
            print "WARNING: More than one class is imported in {} and name is \
                    undefined, could be picking the wrong one".format(path)
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
    def update_dict(yml, D):
        for k,v in yml.iteritems():
            if D.has_key(k) and k != "path":
                raise KeyError("Key collision with {} when ".format(k) + \
                              "trying to create the configuration file. Two keys \
                              with the same name (ex.: two batch_size?) is ambiguous \
                              and I am aborting. (only 'path' is an exception)")
            D.update({k: v})

    def recursive_parse(yml, D):
        if isinstance(yml, dict):
            if yml.has_key('path'):
                for p in assert_iterable_return_iterable(yml['path']):
                    recursive_parse(p, D)
            update_dict(yml, D)
            return

        assert isinstance(yml, str) and '.yml' in yml, "Problem occured in recursive parsing"
        with open(yml, 'r') as f:
            next_yml = yaml.load(f)
        recursive_parse(next_yml, D)

    D = {}
    recursive_parse(configyml, D)
    return D
