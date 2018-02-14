from deepmonster.machinery.runner import TrainModel

from deepmonster.blocks.extensions import (Experiment, SaveExperiment, LoadExperiment,
                                           FrameGen, AdjustSharedVariable, Ipdb)
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring

import theano
from deepmonster.nnet.frameworks.popstats import get_inference_graph

class ClassificationRunner(TrainModel):
    def configure(self):
        self.assert_for_keys(['epochs', 'save_freq'])
        super(ClassificationRunner, self).configure()


    def build_extensions_list(self):
        epochs = self.config['epochs']
        save_freq = self.config['save_freq']

        original_save = self.config['original_save'] if self.config.has_key('original_save') \
                else False

        tracked_train = self.model.linksholder.filter_and_broadcast_request('train', 'TrackingLink')
        tracked_valid = self.model.linksholder.filter_and_broadcast_request('valid', 'TrackingLink')

        extensions = [
            Timing(),
            Experiment(
                self.filemanager.exp_name,
                self.filemanager.local_path,
                self.filemanager.network_path,
                self.filemanager.full_dump),
            FinishAfter(after_n_epochs=epochs),
            TrainingDataMonitoring(
                tracked_train,
                prefix="train",
                after_epoch=True),
            DataStreamMonitoring(
                tracked_valid,
                self.streams['valid'],
                prefix="valid",
                after_epoch=True),
            #Ipdb(after_batch=True),
            SaveExperiment(
                self.model.parameters,
                original_save=original_save,
                every_n_epochs=save_freq),
        ]


        extensions += [
            #LoadExperiment(
            #    parameters,
            #    full_load=False),
            ProgressBar(),
            Printing(),
        ]

        self.extensions = extensions
