import os

_config_default = {
    'backend': 'theano',
    'floatX': 'float32',
    'numpy_random_seed': 4321,
}

class DeepMonsterConfigParser(object):
    def __init__(self):
        for k, v in _config_default.iteritems():
            setattr(self, k , v)

        dmenv = os.getenv('DM_FLAGS')
        if dmenv is not None:
            userconfig = dmenv.split(',')
            userconfig = [tuple(x.split('=')) for x in userconfig]
            assert all(len(x) == 2 for x in userconfig), "config error from DM_FLAGS"

            for k, v in userconfig:
                setattr(self, k, v)
