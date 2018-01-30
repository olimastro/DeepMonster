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
from dicttypes import onlynewkeysdict, merge_all_dicts_with_dict
from filemanager import FileManager
from runner import TrainModel
from yml import YmlParser, YmlFieldNotFoundError, unpack_yml
from deepmonster.utils import assert_iterable_return_iterable, logical_xor

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
        TODO: write this
        -----------------------------------------------

    hatch requires two things two run: config values and an assemble object.
    The assemble object can be the library's default. This one also defines
    what are the essential python objects required to hatch a monster.
    The config values need to be in masteryml['config'].

    hatch works with scripts in deepmonster.machinery.yml to make sense out of
    yml files.
    """
    assemble = Assemble() if AssembleClass is None else AssembleClass()

    loader = YmlParser(path_to_yml, assemble.cores_mandatory_to_init + ['config'])
    loader.load()

    core_instances = assemble.assemble_cores(ymlparser=loader)

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


    @property
    def core_to_format_dict(self):
        if hasattr(self, '_ctfd'):
            return self._ctfd
        else:
            self.set_core_to_format_dict()
            return self._ctfd

    def assemble_cores(self, ymlparser=None, config=None, bypass_format=False,
                       masterdict=None):
        """Assemble all cores for this run of the library. A typical run will be done
        through hatch, but some kwargs can be given for this to be used outside
        of hatch. Note that the error parsings will be less helpful when used
        outside of hatch (bypass_format == True)

        Returns a dictionary of {core_name: core_instance}
        """
        assert logical_xor(ymlparser is not None, bypass_format), \
                "ymlparser is not None XOR bypass_format is True needs to be True"
        if ymlparser is not None and masterdict is not None:
            print "WARNING, Assemble was not given a ymlparser and a masterdict, "+\
                    "the masterdict will be ignored."

        # load config
        if config is None:
            if ymlparser is not None:
                config = unpack_yml(ymlparser.yml['config'])
            else:
                config = unpack_yml(masterdict['config'])

        core_configs = {s: None for s in self.core_pieces}

        if not bypass_format:
            core_classes = {s: None for s in self.core_pieces}

            # load mandatory classes
            for req_core in self.cores_mandatory_to_init:
                core_config, cls = ymlparser.load_ymlentry(req_core)
                core_configs[req_core] = core_config
                core_classes[req_core] = cls

            # load optional Cores
            # options have 4 behaviors from the parser:
                # they return a User defined Class
                # they raise YmlFieldNotFoundError, in case we need to default and have 0 extra config
                # they return a cls None and a config, in case we need to default the cls
                # they raise FormatError, in case an error was encountered while parsing
            for opt_core in self.cores_with_default:
                try:
                    core_config, cls = ymlparser.load_ymlentry(opt_core)
                except YmlFieldNotFoundError:
                    core_config = {}
                    cls = None
                if cls is None:
                    cls = self.core_pieces_init_dict[opt_core]

                core_configs[opt_core] = core_config
                core_classes[opt_core] = cls

        else:
            assert all(v is not None for v in self.core_pieces_init_dict.values()),\
                    "Assemble was asked to bypass_format but some values of its core "+\
                    "pieces dictionary is None, it cannot know what classes to use."
            core_classes = self.core_pieces_init_dict
            if masterdict is None:
                core_configs = {s: {} for s in self.core_pieces}
            else:
                core_configs = {s: masterdict.get(s, {}) for s in self.core_pieces}

        # prepare config for all cores
        # this results in a config dict for every core objects where
        # fields defined under their core name are taken as priority
        # over the global ones in config.
        core_configs = merge_all_dicts_with_dict(core_configs, config)

        # init classes
        core_instances = {s: None for s in self.core_pieces}
        for cls in self.core_pieces:
            core_instances[cls] = core_classes[cls](core_configs[cls])

        for instance in core_instances.values():
            for cd in instance.__core_dependancies__:
                instance.bind_a_core(cd, core_instances[cd])

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
            'datafetcher': DeepMonsterFetcher,
            'filemanager': FileManager,
        }
        return cpid

    #NOTE: NotImplemented any more, it is here for future use
    def set_core_to_format_dict(self):
        raise NotImplementedError("This feature is currently unsupported")
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
    return unpack_yml(configyml, 'path')
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
