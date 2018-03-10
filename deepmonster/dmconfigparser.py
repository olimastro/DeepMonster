import os

class ConfigValue(object):
    def __init__(self, name, default_val, help_message, possible_vals=None):
        self.name = name
        self.value = default_val
        self.val_type = type(default_val)
        self.help_message = help_message
        assert possible_vals is None or isinstance(possible_vals, (list, tuple))
        self.possible_vals = [] if possible_vals is None else possible_vals


#TODO: write config help / error messages
class DeepMonsterConfigParser(object):
    __default_config = {}
    __init_count = 0
    __frozen = False

    def __new__(cls, *args, **kwargs):
        if cls.__init_count > 0:
            raise RuntimeError("Initializing again DeepMonsterConfigParser, unsecure")
        cls.__init_count += 1
        return object.__new__(cls, *args, **kwargs)


    def __init__(self):
        for k, cv in self.__default_config.iteritems():
            v = cv.value
            setattr(self, k , v)

        dmenv = os.getenv('DM_FLAGS')
        if dmenv is not None:
            userconfig = dmenv.split(',')
            userconfig = [tuple(x.split('=')) for x in userconfig]
            assert all(len(x) == 2 for x in userconfig), "config error from DM_FLAGS"

            for k, v in userconfig:
                if not self.__default_config.has_key(k):
                    raise ValueError("Unrecognized configuration value for DeepMonster")
                # this can be a little unsecure in the sense that
                # bool('asdf') == True and won't raise an error
                #TODO: enforce a range of possible values
                v_type = self.__default_config[k].val_type
                possible_v = self.__default_config[k].possible_vals
                v = self.parse(k, v, v_type, possible_v)
                setattr(self, k, v)
        self.__frozen = True


    def __setattr__(self, name, value):
        if self.__frozen:
            raise RuntimeError("Cannot change config after freezing it")
        super(DeepMonsterConfigParser, self).__setattr__(name, value)


    def parse(self, k, v, v_type, possible_v):
        if v_type is bool:
            if v in ['False', 'false']:
                return False
            elif v in ['True', 'true']:
                return True
            raise ValueError(
                "Invalid config '{}' parsing of '{}' that should be bool type".format(k, v))
        if v_type in [float, int]:
            try:
                return v_type(v)
            except ValueError:
                raise ValueError(
                    "Cannot parse config '{}' of '{}' that should be '{}' type".format(k, v, v_type))
        if isinstance(val_type, str):
            if len(possible_v) > 0:
                if not v in possible_v:
                    raise ValueError("Illegal config '{}' value for '{}'. Legal values are {}".format(
                        k, v, possible_v))
            return v


    @classmethod
    def add_default_config_value(cls, confval):
        assert isinstance(confval, ConfigValue)
        cls.__default_config.update({confval.name: confval})



known_backend = ['theano', 'tensorflow', 'pytorch']
cv = ConfigValue('backend', 'theano','todo', known_backend)
DeepMonsterConfigParser.add_default_config_value(cv)

cv = ConfigValue('backend_redirection', True, 'todo')
DeepMonsterConfigParser.add_default_config_value(cv)

supported_floatX = ['float16', 'float32']
cv = ConfigValue('floatX', 'float32', 'todo', supported_floatX)
DeepMonsterConfigParser.add_default_config_value(cv)

cv = ConfigValue('numpy_random_seed', 4321, 'todo')
DeepMonsterConfigParser.add_default_config_value(cv)

cv = ConfigValue('debug', False, 'todo')
DeepMonsterConfigParser.add_default_config_value(cv)
