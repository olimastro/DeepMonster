import numpy as np
from fuel.transformers import Transformer
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme
from random import shuffle


class UpsampleToShape(Transformer):
    def __init__(self, shape, *args, **kwargs):
        kwargs.setdefault('produces_examples', False)
        super(UpsampleToShape, self).__init__(*args, **kwargs)
        self.outshp = shape

    def transform_batch(self, batch):
        data = batch[0]
        inshp = data.shape[-2:]

        n = 0 ; shp = inshp
        while shp < self.outshp:
            n += 1
            shp = shp*(2**n)

        for i in range(n+1).reverse():
            if i == 0:
                #crop time
                data = self.crop(data)
            elif i == 1:
                # bilinear time
                data = self.bilinear_upsampling(data)
            else:
                # gk time
                data = self.gaussiankernel_upsampling(data)

        return [data]+list(batch[1:])


    def crop(self, data):
        # bc01
        new_data = np.empty(data.shape[:3] + self.outshp, dtype=np.float32)

        x = np.random.randint(data.shape[-1]-self.outshp[0])
        y = np.random.randint(data.shape[-2]-self.outshp[1])
        for i in range(new_data.shape[0]) :
            new_data[i,:,:,:] = data[i,:,x:x+self.outshp[0],y:y+self.outshp[1]]

        return new_data


    def bilinear_upsampling(self, data):
        pass

    def gaussiankernel_upsampling(self, data):
        pass



class RemoveDtsetMean(Transformer):
    """
        Removes dataset mean from all members of the dataset. Used in vgg ( *old* no batch norm model)
    """
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('produces_examples', False)
        axis = kwargs.pop('axis', 1)
        super(RemoveDtsetMean, self).__init__(*args, **kwargs)
        mean = 0 ; i = 0.
        tup_found = False
        for i, data in enumerate(self.data_stream.get_epoch_iterator()):
            if not tup_found:
                ndim = data[0].ndim
                tup = range(ndim)
                tup.pop(axis)
                tup = tuple(tup)
                tup_found = True

            mean += data[0].mean(axis=tup)
            i += 1.
        self.mean = mean / i


    def transform_batch(self, batch):
        data = batch[0]
        # TODO: fix this so it works on any ndim of data
        data -= self.mean[None,:,None,None]
        return [data]+list(batch[1:])



class Normalize_min1_1(Transformer):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('produces_examples', False)
        super(Normalize_min1_1, self).__init__(*args, **kwargs)

    def transform_batch(self, batch):
        data = batch[0].astype(np.float32)
        data = (data - 127.5) / 127.5
        return [data]+list(batch[1:])


class InsertLabeledExamples(Transformer):
    def __init__(self, dataset, nb_class, nb_examples, examples_per_class, *args, **kwargs):
        norm01 = kwargs.pop('norm01', True)
        start = kwargs.pop('start', 0)
        super(InsertLabeledExamples, self).__init__(*args, **kwargs)
        self.dataset = dataset
        self.norm01 = norm01
        self.start = start
        self.nb_class = nb_class
        self.nb_examples = nb_examples
        self.total_examples = examples_per_class * nb_class
        self.current_slice_index = 0

        # build a list of indexes on which we will loop with the original stream
        print "Building labeled examples indexes list..."
        print self.norm01
        print self.start
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
            if statistics.sum() == self.total_examples:
                break
            _, targets = next(epitr)
            if i < self.start:
                continue
            for j in range(100):
                if statistics[targets[j]].sum() < examples_per_class:
                    indexes += [i + j]
                    statistics[targets[j]] += 1
        if statistics.sum() != self.total_examples:
            raise RuntimeError("Transformer failed")
        self.indexes = indexes


    def transform_batch(self, batch):
        start = self.current_slice_index
        end = start + self.nb_examples
        if end > len(self.indexes) :
            #readjust to restart at the beginning of the list
            #print "INFO: one epoch done on the labeled examples"
            residual = end - len(self.indexes)
            residual_indexes = self.indexes[start:]
            # shuffle everytime an epoch is done, slight chance you have the same
            # example reappearing in the same batch, but very little chance
            shuffle(self.indexes)
            indexes = residual_indexes + self.indexes[:residual]
            self.current_slice_index = residual
        else :
            indexes = self.indexes[start:end]
            self.current_slice_index += self.nb_examples

        data, targets = self.dataset.get_data(state=None, request=indexes)
        if self.norm01 :
            data = data.astype(np.float32) / 255.
        else:
            data = data.astype(np.float32)
            data = (data - 127.5) / 127.5

        data = np.concatenate([data, batch[0]], axis=0)
        #targets = np.concatenate([targets, batch[1]], axis=0)

        return [data, targets]



class CopyBatch(Transformer):
    def transform_batch(self, batch):
        data = np.concatenate([batch[0], batch[0]], axis=0)

        return [data, batch[1]]



class ReLabelLabels(Transformer):
    def __init__(self, nb_class, *args, **kwargs):
        kwargs.setdefault('produces_examples', False)
        super(ReLabelLabels, self).__init__(*args, **kwargs)
        class_list = range(nb_class)
        shuffle(class_list)
        relabel = {}

        for i, j in enumerate(class_list):
            relabel.update({i : j})

        print "New class mapping is:", relabel

        self.relabel = relabel


    def transform_batch(self, batch):
        targets = batch[1]
        for i in len(targets):
            targets[i] = self.relabel[targets[i]]
        batch[1] = targets

        return batch



if __name__ == '__main__':
    from fuel.datasets import CIFAR10
    from fuel.schemes import ShuffledScheme
    from fuel.streams import DataStream

    dataset = CIFAR10(('train',), sources=('features','targets'), subset=slice(0,45000))

    stream = FilterLabelsTransformer(10, 10,
                                     DataStream(
                                         dataset=dataset,
                                         iteration_scheme=ShuffledScheme(
                                             dataset.num_examples,
                                             200)),
                                     produces_examples=False)

    epitr = stream.get_epoch_iterator()
    out = next(epitr)
    print out[0].shape, out[1].shape
    print out[1][:10]
