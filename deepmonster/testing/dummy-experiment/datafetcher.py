import numpy as np

from deepmonster.machinery.datafetcher import DefaultFetcher

from fuel.schemes import SequentialScheme
from fuel.streams import DataStream
from fuel.datasets import Dataset

class DummyFetcher(DefaultFetcher):
    __knows_how_to_load__ = ['dummy']

    @property
    def stream_loader_info(self):
        rval = {
            'func': create_stream,
            'options': [
                'batch_size',
                'input_dims',
                'another_print']}
        return rval


def create_stream(dataset, batch_size, split=('train',), input_dims=64, another_print='none'):
    print another_print
    dataset = DummyDataset(split, input_dims)
    scheme = SequentialScheme(dataset.num_examples, batch_size)
    stream = DataStream(dataset=dataset, iteration_scheme=scheme)
    return stream



class DummyDataset(Dataset):
    def __init__(self, splits, input_dims=64, sources=('inputs',)):
        self.provides_sources = sources
        self.num_examples = 1000
        self.input_dims = input_dims

    def get_data(self, state=None, request=None):
        data = np.random.random((len(request), self.input_dims))
        return data.astype(np.float32)
