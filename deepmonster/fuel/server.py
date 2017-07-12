from multiprocessing import Process
from fuel.streams import ServerDataStream
from fuel.server import start_server

from streams import create_stream



def server_stream(dataset, split, batch_size, **kwargs):
    """This function is the main utility to use fuel.ServerDataStream in conjunction
       with fuel.server. It takes care of launching the subprocesses and returns
       the stream needed in the main script

       kwargs := kwargs needed for the Dataset and fuel server setup
       You can include a list of simple transformer object to wrap the main stream you
       wish included in the server call. Their constructor has to be argumentless.
    """
    #TODO: allow more than 1 split!
    assert split in ['train', 'valid', 'test'], "split name error or NotImplemented more than 1 split"
    if isinstance(dataset, str):
        sources = kwargs.setdefault('sources', ('features',))
    else:
        sources = dataset.provides_sources

    port = kwargs.pop('port',{
        'train': 5557,
        'valid': 5558,
        'test': 5559,})
    hwm = kwargs.pop('hwm', 10)
    wrapping_streams = kwargs.pop('wrapping_streams', [])

    p = Process(target=start_fuel_server, name='fuel_server',
                args=(dataset, split, batch_size, port[split],
                      hwm, wrapping_streams, kwargs))
    p.daemon = True
    p.start()

    sdstream = ServerDataStream(sources, False, port=port[split], hwm=hwm)

    return sdstream


def start_fuel_server(dataset, split, batch_size, port, hwm, wrapping_streams, kwargs):
    kwargs.update({'split': (split,)})
    stream = create_stream(dataset, batch_size, **kwargs)
    for wrap in wrapping_streams:
        stream = wrap(stream)
    start_server(stream, port=port, hwm=hwm)

