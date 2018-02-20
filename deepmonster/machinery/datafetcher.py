from core import LinkingCore
from dicttypes import dictify_type
from linkers import StreamLink
from deepmonster.fuel.streams import (create_stream, DEFAULT_DATASET, create_ssl_stream,
                                     MultipleStreams)
from deepmonster.utils import assert_iterable_return_iterable, flatten

from fuel.transformers import Rename

class DataFetcher(LinkingCore):
    """This class is the unfortunate layer on top of fuel which is already a layer
    on top of the data loading process. It is required in order to harmonize
    the streams, transformers and dataset all under one class that can be flexible.
    The main loading is done in 'fetch' method and stores the streams in its
    linksholder.

    Note that, when writing your own fetcher, you are not forced to define
    the stream loader outside the scope of the class. This is a historical
    artefact, most streams loader prior to the existance of this class
    were written as a function. DataFetcher therefore has the capacity
    to do both in fetch(). The 'from other code' has priority when checking
    which implementation is present.
    """
    __knows_how_to_load__ = NotImplemented
    __reserved_split_keyword__ = NotImplemented
    __default_splits__ = NotImplemented

    def configure(self):
        if self.__knows_how_to_load__ is not NotImplemented:
            assert self.config['dataset'] in self.__knows_how_to_load__, \
                    "Does not know how to load {}".format(self.config['dataset'])
        self.fetch()


    @property
    def stream_loader_info(self):
        """Should return a dict with 'func' as the function to load the stream and
        'options' as a list of possible strings that are keywords to the loader.
        Use when the loader is a function that comes from another place in the code
        (as opposite of being a method of the class).

        NOTE: We assume the function's signature is func(dataset, batch_size, **kwargs)
        """
        return NotImplemented


    def stream_loader(self, split, stream_only_args=None):
        """Load the stream according to the split.
        Use when the loader is a method that comes from the class.
        (as opposite of being a function from an import).

        stream_only_args is a dict of specific config values for this split
        returns a fuel.Datastream
        """
        raise NotImplementedError("Base class has no loader")


    def fetch(self):
        """There are three ways to fetch streams from the config dictionary
        under the 'dataset' field:
            (In order of priority)
            - 'split' field is present mapping to a dict or a list:
                if dict: its keys are split names and further values specific
                values to the split
                if list: its element are split names.
            - 'train, valid or test' are present as keys, fetching those streams.
              They can map to further values specific to the split.
            - The dict has NO key named 'train, valid or test' in which
              case it will default to __default_splits__

            Any other keys in the dict are treated as global settings
        """
        print "Fetching DataStreams"
        # dict of streams splits to their specific options
        splits_args = {}
        if self.config.has_key('split'):
            splits = self.config['split']
            if isinstance(splits, list):
                splits_args = {s: None for s in splits}
            elif isinstance(splits, dict):
                splits_args = splits
            else:
                raise TypeError("Cannot parse split field for dataset, it has to map\
                                to a list or a dict")

        elif self.__reserved_split_keyword__ is not NotImplemented:
            if any(k in self.__reserved_split_keyword__ for k in self.config.keys()):
                for k in self.config.keys():
                    if k in self.__reserved_split_keyword__:
                        splits_args.update(k)

        elif self.__default_splits__ is not NotImplemented:
            splits_args = {s: None for s in self.__default_splits__}

        else:
            raise RuntimeError("There were no information in the dataset config \
                               to choose splits, and no default splits are defined \
                               for this fetcher. Aborting.")

        # support two ways of having a stream loader
        if self.stream_loader_info is NotImplemented:
            # the class should implement its stream loader as a method
            # and access everything it needs through self.config
            loaded_streams = {s: self.stream_loader(s, v) for s,v in splits_args.iteritems()}
        else:
            # the loader comes from another part of the code
            loaded_streams = {s: self.load_stream_external_func(s, v) \
                              for s,v in splits_args.iteritems()}

        # all splits are loaded, store them in linkers for other Cores to play with
        for s, datastream in loaded_streams.iteritems():
            self.store_links(StreamLink(datastream, name=s))


    def load_stream_external_func(self, s, v):
        dataset = self.config['dataset']

        func = self.stream_loader_info['func']
        options = self.stream_loader_info['options']

        v = self.merge_config_with_priority_args(v, options)
        v.pop('split', None) # it should not exist anyway but just to make sure

        split_redirection = v.pop('split_name_redirect', None)
        if split_redirection is not None:
            v_update = {'split': assert_iterable_return_iterable(split_redirection)}
        else:
            v_update = {'split': assert_iterable_return_iterable(s)}
        v.update(v_update)

        try:
            batch_size = v.pop('batch_size')
        except KeyError:
            raise KeyError("No batch size could be found for stream {} and \
                           split {}".format(dataset, s))

        return func(dataset, batch_size, **v)


class DefaultFetcher(DataFetcher):
    __reserved_split_keyword__ = ['train', 'valid', 'test']
    __default_splits__ = ['train', 'valid']


class DeepMonsterFetcher(DefaultFetcher):
    """Default library fetcher uses the logic written in deepmonster.fuel
    """
    __knows_how_to_load__ = DEFAULT_DATASET

    @property
    def stream_loader_info(self):
        rval = {
            'func': create_stream,
            'options': [
                'batch_size',
                'sources',
                'normalization',
                'load_in_memory',
                'test']}
        return rval


class SslFetcher(DefaultFetcher):
    """Fetcher to do semisupervised learning.

    It will always split one dataset split (such as train) into 2 streams, one containing
    a subset of the original split to use for ssl, and the other with / without targets with
    the full dataset.
    """
    __knows_how_to_load__ = DEFAULT_DATASET

    def stream_loader(self, split, split_args):
        if not split_args.has_key('ssl'):
            import ipdb; ipdb.set_trace()
            print "INFO: No ssl specified for split {}, defaulting to regular loading way".format(
                split)
            return self.load_stream_external_func(split, split_args)

        dataset = self.config['dataset']
        split_args = dictify_type(split_args, dict)

        ssl_args = split_args.pop('ssl')
        print_epoch_done = ssl_args.pop('print_epoch_done', False)

        def popargs(args):
            args.pop('split', None) # it should not exist anyway but just to make sure
            try:
                batch_size = args.pop('batch_size')
            except KeyError:
                raise KeyError("No batch size could be found for stream {} and \
                               split {}".format(dataset, split))
            return batch_size

        ssl_kwargs = ['normalization', 'load_in_memory', 'test', 'nb_class',
                      'examples_per_class', 'examples_start']
        kwargs = ['batch_size', 'sources', 'normalization', 'load_in_memory', 'test']

        ssl_args = self.merge_config_with_priority_args(ssl_args, ssl_kwargs)
        ssl_args.update({'normalization': '01'})
        split_args = self.merge_config_with_priority_args(split_args, kwargs)

        batch_size = popargs(split_args)
        ssl_batch_size = popargs(ssl_args)

        split = assert_iterable_return_iterable(split)
        split_args.update({'split': split})
        ssl_args.update({'split': split})

        stream = create_stream(dataset, batch_size, **split_args)
        ssl_stream = create_ssl_stream(dataset, ssl_batch_size, **ssl_args)

        name_dict = {
            'features': 'ssl_features',
            'targets': 'ssl_targets'}

        ssl_stream = Rename(ssl_stream, name_dict)
        streams = [stream, ssl_stream]
        sources = flatten([s.sources for s in streams])
        rval = MultipleStreams(streams, sources)
        return rval
