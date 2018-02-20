import numpy as np
import fuel.datasets
from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.streams import DataStream
from fuel.transformers import Merge

from transformers import Normalize_01, Normalize_min1_1

DEFAULT_DATASET = ['mnist','cifar10','svhn','celeba']

# NOTE: dataset can be a fuel dataset object, if it is a string it will try to fetch it
def create_stream(dataset, batch_size, split=('train',), sources=('features',),
                  normalization='default', load_in_memory=False, test=False):
    assert normalization in [None, 'default', '01','-1+1']
    #TODO: more split
    assert len(split) == 1, "NotImplemented more than 1 split"
    if isinstance(dataset, str):
        assert dataset in DEFAULT_DATASET, "Does not recognize name to fetch"
        dataset = get_dataset(dataset, split, sources, load_in_memory)

    scheme = get_scheme(split, batch_size, dataset.num_examples, test)
    if normalization == 'default':
        stream = DataStream.default_stream(
            dataset=dataset,
            iteration_scheme=scheme)
    else:
        stream = DataStream(
            dataset=dataset,
            iteration_scheme=scheme)
        stream = normalize_stream(stream, normalization)
    return stream


def get_dataset(dataset, split, sources, load_in_memory):
    get_dataset_map = {
        'mnist' : get_mnist,
        'cifar10' : get_cifar10,
        'svhn' : get_svhn,
        'celeba' : get_celeba,
    }
    dataset = get_dataset_map[dataset](split, sources, load_in_memory)
    return dataset


def get_scheme(split, batch_size, num_examples, test=False):
    Scheme = {
        'train' : ShuffledScheme,
        'valid' : SequentialScheme,
        'test' : SequentialScheme,
    }

    if test:
        print "WARNING: Test flag at create_stream, loading stream with one batch_size"
        num_examples = batch_size
    else:
        num_examples = (num_examples // batch_size) * batch_size
    scheme = Scheme[split[0]](num_examples, batch_size)
    return scheme


def normalize_stream(stream, normalization):
    if normalization == '-1+1':
        stream = Normalize_min1_1(stream)
    elif normalization == '01':
        stream = Normalize_01(stream)
    return stream


def create_ssl_stream(dataset, batch_size, split=('train',), normalization='default',
                      load_in_memory=False, test=False,
                      nb_class=10, examples_per_class=500, examples_start=0):
    sources = ('features', 'targets')
    assert normalization in [None, 'default', '01','-1+1']
    #TODO: more split
    assert len(split) == 1, "NotImplemented more than 1 split"
    if isinstance(dataset, str):
        assert dataset in DEFAULT_DATASET, "Does not recognize name to fetch"
        dataset = get_dataset(dataset, split, sources, load_in_memory)

    num_examples = nb_class * examples_per_class

    scheme = get_scheme(split, batch_size, num_examples, test)
    if normalization == 'default':
        stream = SubSetStream.default_stream(
            dataset, nb_class, examples_per_class, start=examples_start,
            iteration_scheme=scheme)
    else:
        stream = SubSetStream(
            dataset, nb_class, examples_per_class, start=examples_start,
            iteration_scheme=scheme)
        stream = normalize_stream(stream, normalization)
    return stream


def get_mnist(split, sources, load_in_memory):
    from fuel.datasets import MNIST
    if 'test' not in split:
        subset = slice(0, 50000) if 'train' in split else slice(50000, 60000)
        split = ('train',)
    else:
        subset = None
    return MNIST(split, sources=sources, subset=subset, load_in_memory=load_in_memory)


def get_cifar10(split, sources, load_in_memory):
    from fuel.datasets import CIFAR10
    if 'test' not in split:
        subset = slice(0, 45000) if 'train' in split else slice(45000, 50000)
        split = ('train',)
    else:
        subset = None
    return CIFAR10(split, sources=sources, subset=subset, load_in_memory=load_in_memory)


def get_svhn(split, sources, load_in_memory):
    from fuel.datasets import SVHN
    if 'test' not in split:
        subset = slice(0, 62000) if 'train' in split else slice(62000, 72000)
        split = ('train',)
    else:
        subset = None
    return SVHN(2, split, sources=sources, subset=subset, load_in_memory=load_in_memory)


def get_celeba(split, sources, load_in_memory):
    from fuel.datasets import CelebA
    return CelebA('64', split, sources=sources, load_in_memory=load_in_memory)


class SubSetStream(DataStream):
    def __init__(self, dataset, nb_class, examples_per_class, start=0, **kwargs):
        assert dataset.sources == ('features', 'targets')
        super(SubSetStream, self).__init__(dataset, **kwargs)
        total_examples = nb_class * examples_per_class
        # build a list of indexes corresponding to the subset
        print "Building subset indexes list..."
        stream = DataStream(
            dataset=dataset,
            iteration_scheme=SequentialScheme(
                dataset.num_examples,
                100))
        epitr = stream.get_epoch_iterator()

        statistics = np.zeros(nb_class)
        self.statistics = np.zeros(nb_class)
        indexes = []
        for i in range(0, (dataset.num_examples // 100) * 100, 100):
            if statistics.sum() == total_examples:
                break
            _, targets = next(epitr)
            if i < start:
                continue
            for j in range(100):
                if statistics[targets[j]].sum() < examples_per_class:
                    indexes += [i + j]
                    statistics[targets[j]] += 1
        if statistics.sum() != total_examples:
            raise RuntimeError("Transformer failed")
        self.indexes = indexes


    def get_data(self, request=None):
        request = [self.indexes[i] for i in request]
        return super(SubSetStream, self).get_data(request)


class MultipleStreams(Merge):
    """The original fuel Merge class resets all the the streams
       epoch_iterator at the same time when one raise StopIteration.

       This one will define one epoch over the **FIRST** stream
       in the list, but will call get_epoch_iterator on individual
       streams when they are done looping.
    """
    def __init__(self, *args, **kwargs):
        self._print = kwargs.pop('print_other_streams_epoch_done', False)
        super(MultipleStreams, self).__init__(*args, **kwargs)
        self._streams_epitr_done = [False for dum in range(len(self.data_streams))]


    def _reset_epoch_iterator(self):
        for i in range(len(self._streams_epitr_done)):
            if self._streams_epitr_done[i]:
                self.child_epoch_iterators[i] = self.data_streams[i].get_epoch_iterator()
                self._streams_epitr_done[i] = False


    def get_epoch_iterator(self, **kwargs):
        if not hasattr(self, 'child_epoch_iterators'):
            self.child_epoch_iterators = [data_stream.get_epoch_iterator()
                                          for data_stream in self.data_streams]
        else:
            self._reset_epoch_iterator()
        return super(Merge, self).get_epoch_iterator(**kwargs)


    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        result = []
        for i in range(len(self.child_epoch_iterators)):
            try:
                sub_result = next(self.child_epoch_iterators[i])
            except StopIteration:
                self._streams_epitr_done[i] = True
                if i == 0:
                    # normal ending of epoch signal
                    raise StopIteration
                if self._print:
                    print "Epoch done on stream #", i
                self._reset_epoch_iterator()
                sub_result = next(self.child_epoch_iterators[i])
            result.extend(sub_result)
        return tuple(result)


if __name__ == '__main__':
    svhn = get_svhn(('train',), ('features', 'targets'), False)
    stream = SubSetStream(
        svhn,
        10,
        1000,
        iteration_scheme=ShuffledScheme(10 * 1000, 25))
    epitr = stream.get_epoch_iterator()
    data = next(epitr)
    print data[0].shape, data[1].shape
