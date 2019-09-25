import chainer
from chainer import optimizer


_default_hyperparam = optimizer.Hyperparameter()  # # NOQA
_default_hyperparam.lr = 0.5
_default_hyperparam.n_updates = 5
_default_hyperparam.state = 'maintain'


class LookaheadRule(optimizer.UpdateRule):

    def __init__(self, rule, parent_hyperparam=None, lr=None,
                 n_updates=None, state=None):
        super(LookaheadRule, self).__init__(
            parent_hyperparam or _default_hyperparam)
        if lr is not None:
            self.hyperparam.lr = lr
        if n_updates is not None:
            self.hyperparam.n_updates = n_updates
        if state is not None:
            self.hyperparam.state = state
        self._base_rule = rule

    def init_state(self, param):
        device = param.device
        with chainer.using_device(device):
            self.state['slow_param'] = param.array.copy()

    def update_core(self, param):
        self._base_rule.update_core(param)

        hp = self.hyperparam
        if self.t % hp.n_updates == 0:
            # Update both the slow and the fast parameters.
            slow_param = self.state['slow_param']
            slow_param += (param.array - slow_param) * hp.lr
            param.array[...] = slow_param

            # Reset state.
            state = hp.state
            if state == 'maintain':
                pass
            elif state == 'interpolate':
                raise NotImplementedError()
            elif state == 'reset':
                self._base_rule.init_state(param)
            else:
                raise ValueError('Unsupported state policy: {}'.format(state))

    def serialize(self, serializer):
        super(LookaheadRule, self).serialize(serializer)
        self._base_rule.serialize(serializer['base_rule'])

    def _init_states(self, param):
        super(LookaheadRule, self)._init_states(param)

        # Transfer the states of the base rule to the correct device (the
        # device of `param`) on deserialization. Note that `init_state` is not
        # meant to perform any device transfers.
        self._base_rule._init_states(param)


class Lookahead(optimizer.GradientMethod):

    def __init__(self, optimizer, lr=_default_hyperparam.lr,
                 n_updates=_default_hyperparam.n_updates,
                 state=_default_hyperparam.state):
        super(Lookahead, self).__init__()
        self.hyperparam.lr = lr
        self.hyperparam.n_updates = n_updates
        self.hyperparam.state = state
        self._base_optimizer = optimizer

    lr = optimizer.HyperparameterProxy('lr')
    n_updates = optimizer.HyperparameterProxy('n_updates')
    state = optimizer.HyperparameterProxy('state')

    def setup(self, link):
        super(Lookahead, self).setup(link)
        self._base_optimizer.setup(link)

    def create_update_rule(self):
        return LookaheadRule(
            self._base_optimizer.create_update_rule(), self.hyperparam)
