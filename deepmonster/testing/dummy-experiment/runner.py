from deepmonster.machinery.runner import ModelRunner

class DummyRunner(ModelRunner):
    def run(self):
        print self.config['runner_print']
