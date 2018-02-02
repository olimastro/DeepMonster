from collections import OrderedDict

import theano
import blocks.algorithms
from blocks.algorithms import GradientDescent

from deepmonster.machinery.model import Model, graph_defining_method
from deepmonster.machinery.linkers import ParametersLink
from deepmonster.utils import flatten

class TheanoModel(Model):
    def configure(self):
        super(TheanoModel, self).configure()
        self.assert_for_keys(['optimizer', 'optimizer/type', 'optimizer/learning_rate'])
        # lets at least check we can source the optimzer before compiling graphs
        try:
            self.Optimizer = getattr(blocks.algorithms, self.config['optimizer/type'])
        except AttributeError:
            raise ImportError("Could not import {} from blocks.algorithms".format(
                self.config['optimizer/type']))


    def build_bprop_graph(self):
        optimizer = self.get_optimizer()
        costs = self.link_here('costs').values()
        import ipdb; ipdb.set_trace()

        # there are either costs assigned to specific params
        isinstance_check = [isinstance(c, ParametersLink) for c in costs]
        if any(isinstance_check):
            assert all(isinstance_check), "Some costs have parameters associated "+\
                    "to them and others don't. All costs need to be bound."
            grads = OrderedDict()
            for paramlink in costs:
                cost = paramlink.raw_var
                assert len(cost) == 1
                params = flatten([self.architecture[arch].parameters for arch in \
                                  paramlink.architectures] + paramlink.parameters)
                grads.update(zip(params,
                                 theano.grad(cost[0], params)))
            cost = None
        # OR let blocks do the gradient
        else:
            cost = costs[0]
            for c in costs[1:]:
                cost += c
            grads = None

        algorithm = GradientDescent(
            cost=cost, gradients=grads,
            parameters=self.parameters,
            step_rule=optimizer)

        self.algorithm = algorithm


    def get_optimizer(self):
        optimizer_dict = self.config['optimizer'].copy()
        lr = optimizer_dict['learning_rate']
        optimizer_param_dict = {k: optimizer_dict[k] for k \
                                in set(optimizer_dict.keys()) - set(['type', 'learning_rate'])}

        if len(optimizer_param_dict.keys()) > 0:
            optimizer = self.Optimizer(learning_rate=lr, **optimizer_param_dict)
        else:
            optimizer = self.Optimizer(lr)

        return optimizer



if __name__ == "__main__":
    from deepmonster.nnet.simple import FullyConnectedLayer
    from deepmonster.machinery.linkers import TrackingLink
    from blocks.algorithms import Adam
    class TestModel(TheanoModel):
        @graph_defining_method('fprop', ['y', 'cost'])
        def build_fprop_graph(self):
            x = theano.tensor.matrix('x')
            y = self.architectures[0].fprop(x**2)
            cost = theano.tensor.mean(y - 1.)
            self.link_var_to_graph(cost, 'cost', 'costs')
            tl = TrackingLink(cost, which_set='train')
            return locals()

    config = {
        'optimizer': {
            'type': Adam,
            'learning_rate': 1e-2}
    }
    fl = FullyConnectedLayer(input_dims=12, output_dims=24)
    fl.initialize()

    model = TestModel(config, fl)
    model.build_model()
    import ipdb; ipdb.set_trace()
