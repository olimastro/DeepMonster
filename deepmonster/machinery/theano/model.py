import theano
from collections import OrderedDict
from blocks.algorithms import GradientDescent
from deepmonster.machinery.model import Model, graph_defining_method
from deepmonster.machinery.linkers import ParametersLink

class TheanoModel(Model):
    def build_bprop_graph(self):
        optimizer = self.get_optimizer()
        # there are either costs assigned to specific params
        # OR let blocks do the gradient
        costs = self.link_here('costs').keys()

        isinstance_check = [isinstance(c, ParametersLink) for c in costs]
        if any(isinstance_check):
            assert all(isinstance_check), "Some costs have parameters associated "+\
                    "to them and others don't. All costs need to be binded."
            grads = OrderedDict()
            for cost in costs:
                grads.update(zip(cost.parameters,
                                 theano.grad(cost.model_var, cost.params)))
            cost = None
        else:
            cost = sum(costs)
            grads = None

        algorithm = GradientDescent(
            cost=cost, gradients=grads,
            parameters=self.model_parameters,
            step_rule=optimizer)

        self.algorithm = algorithm


    def get_optimizer(self):
        optimizer_dict = self.config['optimizer']
        Optimizer = optimizer_dict['type']
        lr = optimizer_dict['learning_rate']
        optimizer_param_dict = {k: optimizer_dict[k] for k \
                                in set(optimizer_dict.keys()) - set(['type', 'learning_rate'])}

        if len(optimizer_param_dict.keys()) > 0:
            optimizer = Optimizer(lr, **optimizer_param_dict)
        else:
            optimizer = Optimizer(lr)

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
