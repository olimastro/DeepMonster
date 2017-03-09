import theano
from theano import tensor

from blocks.algorithms import Adam as BlocksAdam
from blocks.roles import add_role, ALGORITHM_HYPERPARAMETER, ALGORITHM_BUFFER
from blocks.utils import shared_floatx_zeros_matching, shared_floatx



class Adam(BlocksAdam):
    """
        Same as blocks' Adam but with params with name
    """
    def __init__(self, learning_rate=0.002,
                 beta1=0.1, beta2=0.001, epsilon=1e-8,
                 decay_factor=(1 - 1e-8)):
        if isinstance(learning_rate, theano.compile.SharedVariable):
            self.learning_rate = learning_rate
        else:
            self.learning_rate = shared_floatx(learning_rate, "learning_rate")
        self.beta1 = shared_floatx(beta1, "beta1")
        self.beta2 = shared_floatx(beta2, "beta2")
        self.epsilon = shared_floatx(epsilon, "epsilon")
        self.decay_factor = shared_floatx(decay_factor, "decay_factor")
        for param in [self.learning_rate, self.beta1, self.beta2, self.epsilon,
                      self.decay_factor]:
            add_role(param, ALGORITHM_HYPERPARAMETER)

    def compute_step(self, parameter, previous_step):
        mean = shared_floatx_zeros_matching(parameter, 'mean')
        add_role(mean, ALGORITHM_BUFFER)
        variance = shared_floatx_zeros_matching(parameter, 'variance')
        add_role(variance, ALGORITHM_BUFFER)
        time = shared_floatx(0., 'time')
        add_role(time, ALGORITHM_BUFFER)

        t1 = time + 1
        learning_rate = (self.learning_rate *
                         tensor.sqrt((1. - (1. - self.beta2)**t1)) /
                         (1. - (1. - self.beta1)**t1))
        beta_1t = 1 - (1 - self.beta1) * self.decay_factor ** (t1 - 1)
        mean_t = beta_1t * previous_step + (1. - beta_1t) * mean
        variance_t = (self.beta2 * tensor.sqr(previous_step) +
                      (1. - self.beta2) * variance)
        step = (learning_rate * mean_t /
                (tensor.sqrt(variance_t) + self.epsilon))

        mean.name = 'OPT_'+parameter.name + '_mean'
        variance.name = 'OPT_'+parameter.name + '_variance'
        time.name = 'OPT_'+parameter.name + '_time'

        updates = [(mean, mean_t),
                   (variance, variance_t),
                   (time, t1)]

        return step, updates
