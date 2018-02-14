import numpy as np
import theano
import theano.tensor as T

from deepmonster.machinery.model import graph_defining_method
from deepmonster.machinery.linkers import ParametersLink
from deepmonster.machinery.frameworks import TheanoModel
from deepmonster.utils import assert_iterable_return_iterable


class MnistClassification(TheanoModel):
    def build_fprop_graph(self):
        self.graph_classify()


    @graph_defining_method(['loss'], 'costs')
    def graph_classify(self):
        x = T.ftensor4('features')
        y = T.imatrix('targets')

        # train graph
        preds = self.architecture['classifier'].fprop(x)
        loss = T.nnet.categorical_crossentropy(preds, y.flatten()).mean()
        loss.name = 'cost'
        missclass = T.neq(T.argmax(preds, axis=1), y.flatten()).mean()
        missclass.name = 'missclass'

        self.track_var(loss)
        self.track_var(missclass)

        # this is unecessary, but it is to test the ParametersLink functionality
        loss = ParametersLink(loss, architectures=['classifier'])

        return locals()
