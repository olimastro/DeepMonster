#TODO: modify the mainloop so it is no longer bound only to Theano
from blocks.main_loop import MainLoop

from . import Configurable
from extensions import ExtensionsFactory

class ModelRunner(Configurable):
    """All a model needs in order to be runned is a dataset.
    Anything else, such as running the script in order to train the
    model or only load its parameters and play around with it, should be done
    through a subclass.
    """
    def __init__(self, model, config):
        self.model = model
        super(ModelRunner, self).__init__(config)

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
        """Read-only:
            Any change on the linksholder should be done through Model's
            method calls.
        """
        return model.linksholder

    def run(self):
        raise NotImplementedError("This is the base runner class, cannot run anything")



class TrainModel(ModelRunner):
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

    def configure(self):
        self.extensions = []
        self.specific_extensions = []
        for ext in self.config['extensions']:
            if ext.has_key('disabled'):
                self.excluded_extensions.append(ext)


    def add_extension(self, extension):
        # these should be extensions (initialized) themselves, not
        # strings pointing to ones
        self.specific_extensions.append(extension)

    def exclude_extension(self, extension):
        self.excluded_extensions.append(extension)


    def get_extension(self, ext):
        """Extensions all have their own args and kwargs.
        This function returns the initialized asked extension by ways of
        the factory.
        """
        ext_linkers, ext_config = ExtensionsFactory.filter_configs_linkers(
            ext, self.config, self.linksholder.links)
        return ExtensionsFactory(ext, ext_config, ext_linkers)


    def build_extensions_list(self):
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


    def run(self):
        print "Calling MainLoop"
        main_loop = MainLoop(data_stream=self.streams['mainloop'],
                             algorithm=self.model.algorithm,
                             extensions=self.extensions)
        main_loop.run()
