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
from datafetcher import DeepMonsterFetcher
from dicttypes import onlynewkeysdict, merge_dict_as_type
from runner import TrainModel
from yml import (YmlLoader, YmlFieldNotFoundError, StandardFormat,
                 RecursivelyDefinedFormat, DefaultNamedFormat)
from deepmonster.utils import (merge_dicts, assert_iterable_return_iterable,
                               logical_xor)

# the scripts here are made to be more tight in their error checking
# we want the program to crash when the core is being setup if it finds
# an error as it would be really painfull for errors to silently
# slip by when an experiment is running

#TODO: for future version, hatch should be an assemble method
def hatch(path_to_yml, AssembleClass=None):
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
    The config values need to be in masteryml['config'].

    hatch works with scripts in deepmonster.machinery.yml to make sense out of
    yml files.
    """
    assemble = Assemble() if AssembleClass is None else AssembleClass()

    loader = YmlLoader(path_to_yml, assemble.cores_mandatory_to_init + ['config'])
    loader.load()

    core_instances = assemble.assemble_cores(ymlloader=loader)

    # hatch the monster!
    core_instances[assemble.runnable].run()


class Assemble(object):
    """An assemble object of the deep monster library defines the core pieces that
    are entrypoints when launching the main script. They are child of the Core class
    and typically defined in other python script.

    Assemble's job is to provide a setup of core objects that make the library do
    its purpose. It can actually be anything, and the entry script, hatch, will
    query Assemble for all its Core classes and finish by calling 'run()' on what
    Assemble tells it is runnable.

    In spirit, the Assemble object is the bridge between the functionality of
    deepmosnter and the yml files describing the values it needs in order to run.
    It therefore specifies Core objects and YmlParser objects. Both are then
    used in 'hatch' to create the required deepmonster.
    """
    def __init__(self):
        self.core_pieces = self.core_pieces_init_dict.keys()
        self.cores_with_default = [x for x,v in self.core_pieces_init_dict.iteritems() if v is not None]
        self.cores_mandatory_to_init = list(set(self.core_pieces) - set(self.cores_with_default))

    def get_core_config_mapping(self, corekey):
        """As a design choice, there could be fields in the config
        which are not the correct names in the library for their
        corresponding core.
        """
        return self.core_config_mapping.get(corekey, corekey)

    @property
    def core_to_format_dict(self):
        if hasattr(self, '_ctfd'):
            return self._ctfd
        else:
            self.set_core_to_format_dict()
            return self._ctfd

    def assemble_cores(self, ymlloader=None, config=None, bypass_format=False,
                       masterdict=None):
        """Assemble all cores for this run of the library. A typical run will be done
        through hatch, but some kwargs can be given for this to be used outside
        of hatch. Note that the error parsings will be less helpful when used
        outside of hatch (bypass_format == True)

        Returns a dictionary of {core_name: core_instance}
        """
        assert logical_xor(ymlloader is not None, bypass_format), \
                "ymlloader is not None XOR bypass_format is True needs to be True"
        if ymlloader is not None and masterdict is not None:
            print "WARNING, Assemble was not given a ymlloader and a masterdict, "+\
                    "the masterdict will be ignored."

        # load config
        if config is None:
            if ymlloader is not None:
                config = parse_config(ymlloader.yml['config'])
            else:
                config = parse_config(masterdict['config'])

        if not bypass_format:
            core_classes = {s: None for s in self.core_pieces}

            # load mandatory classes
            for req_core in self.cores_mandatory_to_init:
                format_parser = self.core_to_format_dict[req_core]
                core_classes[req_core] = ymlloader.load_ymlentry(req_core, format_parser)

            # load optional Cores
            # options have 4 behaviors from the loader:
                # they return a User defined Class
                # they raise YmlFieldNotFoundError, in case we need to default
                # they return None, in case the format was good AND we need to default
                # they raise FormatError, in case an error was encountered while parsing
            for opt_core in self.cores_with_default:
                format_parser = self.core_to_format_dict[opt_core]
                try:
                    rval = ymlloader.load_ymlentry(opt_core, format_parser)
                except YmlFieldNotFoundError:
                    rval = None
                if rval is None:
                    rval = self.core_pieces_init_dict[opt_core]

                core_classes[opt_core] = rval

            # this results in a config dict for every core objects where
            # fields defined under their core name are taken as priority
            # over the global ones in config.
            core_configs = {
                s: merge_dict_as_type(config, ymlload.yml.get(s, {})) for s in self.core_pieces}

        else:
            assert all(v is not None for v in self.core_pieces_init_dict.values()),\
                    "Assemble was asked to bypass_format but some values of its core "+\
                    "pieces dictionary is None, it cannot know what classes to use."
            core_classes = self.core_pieces_init_dict
            if masterdict is None:
                core_configs = {s: config for s in self.core_pieces}
            else:
                core_configs = {
                    s: merge_dict_as_type(config, masterdict.get(s, {})) for s in self.core_pieces}

        # init classes
        core_instances = {s: None for s in self.core_pieces}
        for cls in self.core_pieces:
            core_instances[cls] = core_classes[cls](core_configs[cls])

        for instance in core_instances.values():
            for cd in instance.__core_dependancies__:
                ci_key = self.get_core_config_mapping(cd)
                instance.bind_a_core(cd, core_instances[ci_key])

        return core_instances


    ### When subclassing Assemble, these below can be changed 'easily' without breaking hatch
    @property
    def core_pieces_init_dict(self):
        """Core classes to assemble in order to make the library runnable
        Map to None for mandatory class to source in some file
        or map to a default if none is provided
        """
        cpid = {
            'architecture': None,
            'model': None,
            'runner': TrainModel,
            'dataset': DeepMonsterFetcher,
        }
        return cpid

    def set_core_to_format_dict(self):
        """Mapping from core classes to the YmlParsing format they can
        ingest and understand to be loaded and setup without causing errors
        """
        ctfd = {
            'architecture': StandardFormat('architecture'),
            'model': StandardFormat('model'),
            'runner': StandardFormat('runner'),
            'dataset': DefaultNamedFormat(
                'dataset', DeepMonsterFetcher.__knows_how_to_load__),
        }
        self._ctfd = ctfd

    @property
    def core_config_mapping(self):
        """Dict from a core name to a config name
        """
        return {'datafetcher': 'dataset'}

    @property
    def runnable(self):
        """Runnable core that can launch what all these cores have been
        assembled for.
        """
        return 'runner'


def parse_config(configyml):
    """Parse the configuration file. The argument here is either a string
    pointing to a yml, or a dict with config values and potentially
    a 'path' key pointing to other yml to join. It will raise an error
    if two keys have the same name (except for path).

    Returns a merge of all the config files as the final configuration dictionnary.
    """
    format_parser = RecursivelyDefinedFormat('config', '.yml')
    def update_dict(yml, D):
        _yml = yml.copy()
        _yml.pop('path', None)
        D.update(_yml)

    def recursive_parse(yml, D):
        format_parser.check(yml)
        if isinstance(yml, dict):
            if yml.has_key('path'):
                for p in assert_iterable_return_iterable(yml['path']):
                    recursive_parse(p, D)
            update_dict(yml, D)
            return

        next_yml = YmlLoader.load_from_path(yml)
        recursive_parse(next_yml, D)

    D = onlynewkeysdict({})
    recursive_parse(configyml, D)
    return D
