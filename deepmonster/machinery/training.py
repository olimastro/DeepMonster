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
from blocks.main_loop import MainLoop

from extensions import ExtensionsFactory

# The order in the mainloop IS important
# user defined exp will typically go in the middle
# of these two blocks
default_extensions_before = [
    'Timing',
    'FinishAfter',
    'Experiment',
    'TrainingDataMonitoring',
    'LogAndSaveStuff',
    'SaveExperiment',
]
default_extensions_after = [
    'ProgressBar',
    'Printing',
]
default_extensions = default_extensions_before + default_extensions_after


class RunModel(object):
    def __init__(self, model, specific_extensions=[]):
        self.model = model
        self.config = model.config
        self.specific_extensions = specific_extensions
        self.excluded_extensions = []
        self.extensions = []

    @property
    def streams(self):
        sl = self.linksholder.filter_linkers('StreamLink')
        sd = {}
        if len(links) == 1:
            sd = {'mainloop': sl.links[0].stream}
        else:
            for link in sl:
                if not sd.has_key('mainloop') and \
                   (link.name == 'mainloop' or link.name == 'train'):
                    sd.update({'mainloop': link.stream})
                else:
                    sd.update({link.name: sl.links.stream})
        return sd

    @property
    def linksholder(self):
        """Any change on the linksholder should be done through Model's
        method calls.
        """
        return model.linksholder


    def add_extension(self, extension):
        # these should be extensions (initialized) themselves, not
        # strings pointing to ones
        self.specific_extensions.append(extension)

    def exclude_extension(self, extension):
        self.excluded_extensions.append(extension)

    def exclude_extensions_from_config(self):
        for ext in self.config['extensions']:
            if ext.has_key('disabled'):
                self.excluded_extensions.append(ext)


    def get_extension(self, ext):
        """Extensions all have their own args and kwargs.
        This function returns the initialized asked extension by ways of
        the factory.
        """
        ext_linkers, ext_config = ExtensionsFactory.filter_configs_linkers(
            ext, self.config, self.linksholder.links)
        return ExtensionsFactory(ext, ext_config, ext_linkers)


    def build_extensions_list(self):
        self.exclude_extensions_from_config()
        def build_list(ext_list):
            extensions_str = [ext for ext in ext_list if ext not in self.excluded_extensions]
            extensions_obj = []
            for ext in extensions_str:
                extensions_obj.append(self.get_extension(ext))
            return extensions_obj

        self.extensions.extend(build_list(default_extensions_before))
        self.extensions.extend(self.specific_extensions)
        self.extensions.extend(build_list(default_extensions_after))


    def add_stream(self, stream, fuel_stream):
        self.model.add_stream(stream, fuel_stream)


    def train(self):
        print "Calling MainLoop"
        main_loop = MainLoop(data_stream=self.streams['mainloop'],
                             algorithm=self.model.algorithm,
                             extensions=self.extensions)
        main_loop.run()




