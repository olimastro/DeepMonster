from multiprocessing import Process
from fuel.streams import ServerDataStream
from fuel.server import start_server

from streams import create_stream



def server_stream(dataset_dict=None, stream=None, hwm=10, wrapping_streams=[],
                  port=dict(train=5557, valid=5558, test=5559), **kwargs):
    """This function is the main utility to use fuel.ServerDataStream in conjunction
       with fuel.server. It takes care of launching the subprocesses and returns
       the stream needed in the main script

       ***This function has 2 different mutually exclusive modes:
       1) dataset_dict is not None := it will start a stream and do all the background work
           to instanciate the dataset object and the appropriate streams (mostly done by create_stream)
       2) stream is not None := it will use this stream as is

       port := port for server
       hwm := high-water mark
       wrapping_streams (only used if dataset is not None) :=
           You can include a list of simple transformer object to wrap the main stream you
           wish included in the server call. Their constructor has to be argumentless.
       kwargs := kwargs needed for the Dataset and fuel server setup
    """
    a = dataset_dict is None
    b = stream is None
    assert (a and not b) or (not a and b), "Specify a dataset XOR a stream"

    if stream is None:
        assert isinstance(dataset_dict, dict) and len(dataset_dict.keys()) == 3, \
                "dataset_dict needs to be a dict "+\
                "with keys [dataset (str name or fuel object), split, batch_size]"
        dataset = dataset_dict['dataset']
        split = dataset_dict['split']
        batch_size = dataset_dict['batch_size']

        #TODO: allow more than 1 split!
        assert split in ['train', 'valid', 'test'], \
                "split name error or NotImplemented more than 1 split"

        # auto-assign port depending on the split
        port = port[split] if isinstance(port, dict) else port

        if isinstance(dataset, str):
            sources = kwargs.setdefault('sources', ('features',))
        else:
            sources = dataset.provides_sources
        p = Process(target=create_stream_and_start_server, name='fuel_server',
                    args=(dataset, split, batch_size, port,
                          hwm, wrapping_streams, kwargs))
    else:
        sources = stream.sources
        p = Process(target=start_fuel_server, name='fuel_server',
                    args=(stream, port, hwm))

    p.daemon = True
    p.start()

    sdstream = ServerDataStream(sources, False, port=port, hwm=hwm)

    return sdstream


def create_stream_and_start_server(dataset, split, batch_size, port, hwm, wrapping_streams, kwargs):
    kwargs.update({'split': (split,)})
    stream = create_stream(dataset, batch_size, **kwargs)
    for wrap in wrapping_streams:
        stream = wrap(stream)
    start_server(stream, port=port, hwm=hwm)


def start_fuel_server(stream, port, hwm):
    start_server(stream, port=port, hwm=hwm)
