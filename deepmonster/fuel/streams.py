from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.streams import DataStream

from rali.transformer import InsertLabeledExamples, CopyBatch, Normalize_min1_1


def create_stream(dataset, batch_size, split=('train',), sources=('features',),
                  normalization='01', ssl={}, load_in_memory=False):
    get_dataset = {
        'mnist' : get_mnist,
        'cifar10' : get_cifar10,
        'svhn' : get_svhn,
    }

    Scheme = {
        'train' : ShuffledScheme,
        'valid' : SequentialScheme,
        'test' : SequentialScheme,
    }

    assert normalization in ['01','-1+1']
    assert dataset in ['mnist','cifar10','svhn']
    assert len(split) == 1

    dataset = get_dataset[dataset](split, sources, load_in_memory)
    scheme = Scheme[split[0]]((dataset.num_examples // batch_size) * batch_size,
                              batch_size)

    if normalization is '01':
        stream = DataStream.default_stream(
            dataset=dataset,
            iteration_scheme=scheme)
    else:
        stream = Normalize_min1_1(
            DataStream(
                dataset=dataset,
                iteration_scheme=scheme))

    if len(ssl) > 0:
        raise NotImplementedError('didnt implement ssl stream fetcher')

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
    return SVHN(2, split, sources=sources, load_in_memory=load_in_memory)


# ssl
"""
train_dataset = CIFAR10(('train',), sources=('features','targets',), subset=slice(0, 45000))
valid_dataset = CIFAR10(('train',), sources=('features','targets',), subset=slice(45000, 50000))
test_dataset = CIFAR10(('test',), sources=('features','targets',))

main_loop_stream = InsertLabeledExamples(
    train_dataset, nb_class, batch_size_supervised, examples_per_class,
    Normalize_min1_1(
        DataStream(
            dataset=train_dataset,
            iteration_scheme=ShuffledScheme(
                (train_dataset.num_examples // batch_size) * batch_size, batch_size))),
                #batch_size, batch_size))),
    norm01=False,
    start = int(tag) * 4000,
    produces_examples=False)

valid_stream = CopyBatch(
    Normalize_min1_1(
        DataStream(
            dataset=valid_dataset,
            iteration_scheme=ShuffledScheme(
                (valid_dataset.num_examples // valid_monitoring_batch_size) * \
                 valid_monitoring_batch_size, valid_monitoring_batch_size))),
    produces_examples=False)

test_stream = CopyBatch(
    Normalize_min1_1(
        DataStream(
            dataset=test_dataset,
            iteration_scheme=ShuffledScheme(
                (test_dataset.num_examples // valid_monitoring_batch_size) * \
                 valid_monitoring_batch_size, valid_monitoring_batch_size))),
    produces_examples=False)
"""

if __name__ == '__main__':
    import ipdb; ipdb.set_trace()
