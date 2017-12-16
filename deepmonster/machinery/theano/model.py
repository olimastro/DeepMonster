import theano
from collections import OrderedDict
from blocks.algorithms import GradientDescent
from deepmonster.machinery.model import Model

class TheanoModel(Model):
    def build_bprop_graph(self):
        optimizer = self._get_optimizer()
        # there are either costs assigned to specific params
        # OR let blocks do the gradient
        costs = self.outputs_container['cost']

        if any([hasattr(c, 'parameters')]):
            grads = OrderedDict()
            for cost in costs:
                grads.update(zip(cost.params,
                                 theano.grad(cost.var, cost.params)))
            cost = None
        else:
            cost = sum([c.var for c in costs])
            grads = None

        algorithm = GradientDescent(
            cost=cost, gradients=grads,
            parameters=self.containers['parameters'],
            step_rule=optimizer)

        self.algorithm = algorithm


    def _get_optimizer(self):
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
    class TestModel(TheanoModel):
        def build_fprop_graph(self):
            x = theano.tensor.matrix('x')
            y = x**2
            self.link_var_to_graph(y, 'y', 'test')

    model = TestModel([], [])
    model.build_fprop_graph()
    import ipdb; ipdb.set_trace()
