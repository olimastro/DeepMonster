import logging
import itertools
import numpy as np
from collections import OrderedDict
from collections import Mapping
from picklable_itertools.extras import equizip

import theano
from blocks.algorithms import GradientDescent, UpdatesAlgorithm
from blocks.graph import ComputationGraph
from blocks.theano_expressions import l2_norm
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



class EntropySGD(UpdatesAlgorithm):
    """
        Copy Pasta of __init__ method of GradientDescent. Watch out if a block's
        update change this. Unfortunatly, it seems to implement this we need to
        intercept some things happening in the __init__ and cannot directly subclass
        GradientDescent.
    """
    def __init__(self, langevin_itr, eta=0.1, alpha=0.75, gamma0=1e-4,
                 hard_limit_on_scopping=True, scoping=1.001, epsilon=1e-4,
                 cost=None, parameters=None, step_rule=None,
                 gradients=None, known_grads=None, consider_constant=None,
                 **kwargs):
        self.eta = np.float32(eta)
        self.alpha = np.float32(alpha)
        self.gamma = theano.shared(np.float32(gamma0), name='ESGD_gamma')
        self.scoping = np.float32(scoping)
        self.hard_limit_on_scopping = hard_limit_on_scopping
        self.epsilon = np.float32(epsilon)
        self.langevin_itr = langevin_itr
        self.langevin_step = 1

        # Set initial values for cost, parameters, gradients.
        self.cost = cost
        self.parameters = parameters
        # Coerce lists of tuples to OrderedDict. Do not coerce Mappings,
        # as we don't want to convert dict -> OrderedDict and give it
        # an arbitrary, non-deterministic order.
        if gradients is not None and not isinstance(gradients, Mapping):
            gradients = OrderedDict(gradients)
        self.gradients = gradients

        # If we don't have gradients, we'll need to infer them from the
        # cost and the parameters, both of which must not be None.
        if not self.gradients:
            self.gradients = self._compute_gradients(known_grads,
                                                     consider_constant)
        else:
            if cost is not None:
                logger.warning(('{}: gradients already specified directly; '
                                'cost is unused.'
                                .format(self.__class__.__name__)))
            if self.parameters is None and isinstance(gradients, OrderedDict):
                # If the dictionary is ordered, it's safe to use the keys
                # as they have a deterministic order.
                self.parameters = list(self.gradients.keys())
            elif self.parameters is not None:
                # If parameters and gradients.keys() don't match we can
                # try to recover if gradients is ordered.
                if set(self.parameters) != set(self.gradients.keys()):
                    logger.warn("Specified parameters list does not match "
                                "keys in provided gradient dictionary; "
                                "using parameters inferred from gradients")
                    if not isinstance(self.gradients, OrderedDict):
                        raise ValueError(determinism_error)
                    self.parameters = list(self.gradients.keys())
            else:
                # self.parameters is not None, and gradients isn't
                # an OrderedDict. We can't do anything safe.
                raise ValueError(determinism_error)
            if known_grads:
                raise ValueError("known_grads has no effect when gradients "
                                 "are passed in")
            if consider_constant is not None:
                raise ValueError("consider_constant has no effect when "
                                 "gradients are passed in")

        # ------------ESGD interception! -----------------
        # recreating a two list of parameters of theano shared
        # they are x_prime and mu in the paper
        true_parameters = []
        mu_parameters = []
        for param in self.parameters:
            new_param = theano.shared(param.get_value(),
                                      name=param.name)
            # same thing but we need a unique object
            mu_param = theano.shared(param.get_value(),
                                     name=param.name)
            true_parameters += [new_param]
            mu_parameters += [mu_param]

        self.true_parameters = true_parameters
        self.mu_parameters = mu_parameters

        new_gradients = OrderedDict()
        #import ipdb; ipdb.set_trace()
        for true_param, param in zip(true_parameters, self.parameters):
            gradient = self.gradients[param]
            new_gradient = gradient - self.gamma * (true_param - param)
            new_gradients.update({param: new_gradient})
        # gradients now contain the ESGD step (line 4 algo 1 of the paper)
        del self.gradients
        self.gradients = new_gradients


        # The order in which the different gradient terms appears
        # here matters, as floating point addition is non-commutative (and
        # Theano's graph optimizations are not order-independent).
        # This is why we do not use .values().
        gradient_values = [self.gradients[p] for p in self.parameters]
        self.total_gradient_norm = (l2_norm(gradient_values)
                                    .copy(name="total_gradient_norm"))

        self.step_rule = step_rule if step_rule else Scale()


        logger.debug("Computing parameter steps...")
        self.steps, self.step_rule_updates = (
            self.step_rule.compute_steps(self.gradients))

        # Same as gradient_values above: the order may influence a
        # bunch of things, so enforce a consistent one (don't use
        # .values()).
        step_values = [self.steps[p] for p in self.parameters]
        self.total_step_norm = (l2_norm(step_values)
                                .copy(name="total_step_norm"))

        # Once again, iterating on gradients may not be deterministically
        # ordered if it is not an OrderedDict. We add the updates here in
        # the order specified in self.parameters. Keep it this way to
        # maintain reproducibility.

        # ---- Another ESGD interception here! -----------------
        randrg = theano.tensor.shared_randomstreams.RandomStreams(seed=1234)
        eps = self.epsilon * randrg.normal(dtype=theano.config.floatX)
        eta_prime = getattr(self.step_rule, 'learning_rate')
        slgd_eta_update = theano.tensor.sqrt(eta_prime) * eps

        kwargs.setdefault('updates', []).extend(
            itertools.chain(((parameter, parameter - self.steps[parameter] + slgd_eta_update)
                             for parameter in self.parameters),
                            self.step_rule_updates)
        )

        mu_updates = [(mu, np.float32(1. - self.alpha) * mu + self.alpha * x_prime) for mu, x_prime \
                      in zip(self.mu_parameters, self.parameters)]
        self.mu_updates = mu_updates

        super(EntropySGD, self).__init__(**kwargs)
        #import ipdb; ipdb.set_trace()


    def initialize(self):
        logger.info("Initializing the training algorithm")
        update_values = [new_value for _, new_value in self.updates]
        logger.debug("Inferring graph inputs...")
        try:
             self.inputs = ComputationGraph(update_values).inputs
        except AttributeError:
            print "oh you silly blocks, why do you even?"
        logger.debug("Compiling training function...")

        # line 5
        try:
            self._function_for_x_prime = theano.function(
                self.inputs, [], updates=self.updates, **self.theano_func_kwargs)
        except TypeError as e:
            print "This following error was thrown: ", e
            print "Let's try to fix broadcastable patterns and recompile..."
            # this is ugly, but lets face the ugliness
            for i, tup in enumerate(self.updates):
                var = tup[0]
                updt = tup[1]
                if updt.broadcastable != var.broadcastable:
                    import ipdb; ipdb.set_trace()
                    updt = theano.tensor.patternbroadcast(updt, var.broadcastable)
                    self.updates[i] = (var, updt)
            self._function_for_x_prime = theano.function(
                self.inputs, [], updates=self.updates, **self.theano_func_kwargs)
        # line 6
        self._function_for_mu = theano.function(
            [], [], updates=self.mu_updates)

        logger.info("The training algorithm is initialized")


    def process_batch(self, batch):
        self._validate_source_names(batch)
        ordered_batch = [batch[v.name] for v in self.inputs]

        if self.langevin_step % self.langevin_itr == 0:
            # new langevin epoch
            # update the true param and reassign x_prime and mu with the true param values
            for param, mu, true_param in zip(self.parameters, self.mu_parameters,
                                             self.true_parameters):
                true_param.set_value(true_param.get_value() - self.eta * (true_param.get_value() - \
                                                                          mu.get_value()))
                param.set_value(true_param.get_value())
                mu.set_value(true_param.get_value())

            gamma_val = self.gamma.get_value()
            if gamma_val > 1. and self.hard_limit_on_scopping:
                gamma_val = np.float32(1.)
            self.gamma.set_value(gamma_val * self.scoping)
            self.langevin_step = 0

        self._function_for_x_prime(*ordered_batch)
        self._function_for_mu()
        self.langevin_step += 1


    def _compute_gradients(self, known_grads, consider_constant):
        if self.cost is None:
            raise ValueError("can't infer gradients; no cost specified")
        elif self.parameters is None or len(self.parameters) == 0:
            raise ValueError("can't infer gradients; no parameters "
                             "specified")
        # While this strictly speaking could be a dict and not an
        # OrderedDict (because we iterate over it in the order of
        # self.parameters), this guards a little bit against
        # nondeterminism introduced by future refactoring.
        logger.info("Taking the cost gradient")
        gradients = OrderedDict(
            equizip(self.parameters, theano.tensor.grad(
                self.cost, self.parameters,
                known_grads=known_grads,
                consider_constant=consider_constant)))
        logger.info("The cost gradient computation graph is built")
        return gradients


if __name__ == '__main__':
    import theano.tensor as T
    import numpy as np
    x = T.fmatrix('x')
    u = T.fmatrix('u')
    npw = np.random.random((64,128)).astype(np.float32)
    w = theano.shared(npw, name='w')

    y = T.dot(x, w)
    cost = T.sqrt(T.sum(y**2))

    updt = [(w, w-(u-w)*(cost/1000.))]

    f = theano.function(inputs=[x, u],outputs=[cost],updates=updt)
    out = f(np.random.random((100,64)).astype(np.float32))
