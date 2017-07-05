from multiprocessing import Process
from fuel.streams import ServerDataStream
from fuel.server import start_server

from streams import create_stream



def server_stream(dataset, split, batch_size, **kwargs):
    """This function is the main utility to use fuel.ServerDataStream in conjunction
       with fuel.server. It takes care of launching the subprocesses and returns
       the stream needed in the main script

       kwargs := kwargs needed for the Dataset and fuel server setup
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

    p = Process(target=start_fuel_server, name='fuel_server',
                args=(dataset, split, batch_size, port[split], hwm, kwargs))
    p.daemon = True
    p.start()

    sdstream = ServerDataStream(sources, False, port=port[split], hwm=hwm)

    return sdstream


def start_fuel_server(dataset, split, batch_size, port, hwm, kwargs):
    kwargs.update({'split': (split,)})
    start_server(create_stream(dataset, batch_size, **kwargs), port=port, hwm=hwm)
