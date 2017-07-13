import numpy as np
from fuel.transformers import Transformer
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme
from random import shuffle



class DefaultTransformer(Transformer):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('produces_examples', False)
        super(DefaultTransformer, self).__init__(*args, **kwargs)
        self.messaged = False

    def trigger_warning(self, msg):
        if not self.messaged:
            print "WARNING: {}: {}".format(self.__class__, msg)
            self.messaged = True


class UpsampleToShape(DefaultTransformer):
    def __init__(self, shape, *args, **kwargs):
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


class Float32(DefaultTransformer):
    def transform_batch(self, batch):
        data = batch[0].astype(np.float32)
        return [data]+list(batch[1:])


class RemoveDtsetMean(DefaultTransformer):
    """
        Removes dataset mean from all members of the dataset. Used in vgg ( *old* no batch norm model)
    """
    def __init__(self, *args, **kwargs):
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


class From01tomin11(DefaultTransformer):
    # 01 to -1+1
    def transform_batch(self, batch):
        data = batch[0]
        data = data*2. - 1.
        return [data]+list(batch[1:])


class AssertDataType(DefaultTransformer):
    def __init__(self, *args, **kwargs):
        super(AssertDataType, self).__init__(*arg, **kwargs)
        data = next(self.data_stream.get_epoch_iterator())[0]
        assert data.dtype == np.uint8, "Cannot use transformer, dtype is not uint8"


class Normalize_min1_1(AssertDataType):
    def transform_batch(self, batch):
        data = batch[0].astype(np.float32)
        data = (data - 127.5) / 127.5
        return [data]+list(batch[1:])


class Normalize_01(AssertDataType):
    def transform_batch(self, batch):
        data = batch[0].astype(np.float32)
        data = data / 255.
        return [data]+list(batch[1:])


class OpticalFlow(DefaultTransformer):
    def __init__(self, *args, **kwargs):
        from cv2 import calcOpticalFlowFarneback as computeof
        self.computeof = computeof
        two_stream_format = kwargs.pop('two_stream_format', 10)
        super(OpticalFlow, self).__init__(*args, **kwargs)

        if two_stream_format != False:
            assert isinstance(two_stream_format, int), "two_stream_format is not False, "+\
                    "expecting an int for amount of flow to compute from a random frame"
        else:
            #TODO:
            raise NotImplementedError("Did not implement another format than the two-stream one")
        self.two_stream_format = two_stream_format
        sources = self.sources
        assert sources[0] == 'features'
        self.sources = (sources[0],) + ('flow',) + sources[1:]


    def transform_batch(self, batch):
        data = batch[0]
        if data.ndim != 5:
            raise TypeError("OpticalFlow wants tbc01 and got shape of {}".format(data.shape))

        if not 'float' in str(data.dtype):
            data = data.astype(np.float32)

        # infer if it is 01 or -1+1 norm
        # bring it back to 255
        if np.min(data) < 0.:
            data = (data*2. - 1.) * 255.
        elif 'int' in str(batch[0].dtype):
            # treat int as if it were a uint
            data = data / 255.
        elif np.max(abs(data)) > 1.:
            # this should not even happen
            data = 255. * (data / np.max(abs(data)))
            msg = "found a wierd normalization, will try to put "+\
                    "it in [0, 255] range but not sure the flow will be right"
            self.trigger_warning(msg)

        # take mean and uintify, tbc01, flow wants grayscale
        data = data.mean(axis=2).astype('uint8')

        if self.two_stream_format != False:
            flows = np.empty((data.shape[1], data.shape[2],
                             data.shape[3], 2 * self.two_stream_format), dtype=np.float32)
            new_data = np.empty(batch[0].shape[1:], dtype=np.float32)
            for b in range(data.shape[1]):
                rf_id = np.random.randint(0, data.shape[0] - self.two_stream_format)
                new_data[b] = batch[0][rf_id,b,...]

                for i in range(0, 2 * self.two_stream_format, 2):
                    flow = self.computeof(data[i,b], data[i+1,b], None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    flows[b,...,i:i+2] = flow

            return [new_data, flows.transpose(0,3,1,2)] + list(batch[1:])


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
