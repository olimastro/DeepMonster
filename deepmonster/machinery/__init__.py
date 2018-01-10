class Configurable(object):
    """Base class for any configurable objects.
    The rationale is that fundamental objects of the
    library, such as Model, Architecture and Runner,
    have their behaviors described by the user in a
    yaml file which is parsed into a python dict.

    The Configurable interface then strive to serve this purpose:
        The entrypoint of any experiment, Launch, which setup
        these objects, doesn't know all the particularities
        required to instanciate a user defined Model for example.
        We therefore need a standard interface to achieve this result.
    """
    def __init__(self, config):
        assert isinstance(config, dict), "Configuration file needs to be a dict"
        self.config = config
        self.configure()


    def configure(self):
        """Define actions on how to configure this object according to the config.
        """
        pass
