import logging
import theano
from blocks.algorithms import GradientDescent
from blocks.graph import ComputationGraph
logger = logging.getLogger(__name__)

class TwoStepGradientDescent(GradientDescent):
    """
        This class takes two groups of names and assigns each update with
        names in the corresponding group to its corresponding theano function.
        Two functions are now runned at process_batch time. They will each be
        runned a number of time stored in grous_steps

        WARNING: This means the parameters need to have names!!!
    """
    def __init__(self, groups={}, groups_steps=[1,1], **kwargs):
        assert len(groups) == 2
        assert len(groups_steps) == 2
        assert all([key in ['group1', 'group2'] for key in groups.keys()])
        self.groups = groups
        self.groups_steps = groups_steps
        super(TwoStepGradientDescent, self).__init__(**kwargs)


    def initialize(self):
        logger.info("Initializing the training algorithm")
        update_values = [new_value for _, new_value in self.updates]
        logger.debug("Inferring graph inputs...")
        try:
             self.inputs = ComputationGraph(update_values).inputs
        except AttributeError:
            print "oh you silly blocks, why do you even?"
        logger.debug("Compiling training function...")
        # find updates according to names in the two groups
        updt_group1 = [x for x in self.updates \
                       if any([name in x[0].name for name in self.groups['group1']])]
        updt_group2 = [x for x in self.updates \
                       if any([name in x[0].name for name in self.groups['group2']])]

        for updt in updt_group1 + updt_group2:
            self.updates.pop(self.updates.index(updt))
        updt_group1 += self.updates
        updt_group2 += self.updates

        self._function1 = theano.function(
            self.inputs, [], updates=updt_group1, **self.theano_func_kwargs)
        self._function2 = theano.function(
            self.inputs, [], updates=updt_group2, **self.theano_func_kwargs)

        logger.info("The training algorithm is initialized")


    def process_batch(self, batch):
        self._validate_source_names(batch)
        ordered_batch = [batch[v.name] for v in self.inputs]
        for i in range(self.groups_steps[0]):
            self._function1(*ordered_batch)
        for i in range(self.groups_steps[1]):
            self._function2(*ordered_batch)
