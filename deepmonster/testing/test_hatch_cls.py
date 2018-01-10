from deepmonster.machinery import Configurable
from deepmonster.machinery.model import Model
from deepmonster.machinery.runner import ModelRunner

class DummyArch(Configurable):
    pass


class DummyModel(Model):
    pass


class DummyRunner(ModelRunner):
    pass
    def run():
        print "Hello!"
