import types

import chainer
from chainer import optimizer


_default_hyperparam = optimizer.Hyperparameter()  # # NOQA
_default_hyperparam.lr = 0.5
_default_hyperparam.n_updates = 5
_default_hyperparam.state = 'maintain'


def _to_lookahead_init_state(init_state):
    def lookahead_init_state(self, param):
        init_state(param)
        with chainer.using_device(param.device):
            self.state['lookahead_slow_param'] = param.array.copy()

    return lookahead_init_state


def _to_lookahead_update_core(update_core):
    def lookahead_update_core(self, param):
        update_core(param)
        hp = self.hyperparam
        if self.t % hp.lookahead_n_updates == 0:
            # Update both the slow and the fast parameters.
            slow_param = self.state['lookahead_slow_param']
            slow_param += (param.array - slow_param) * hp.lookahead_lr
            param.array[...] = slow_param

            # Reset state.
            state = hp.lookahead_state
            if state == 'maintain':
                pass
            elif state == 'interpolate':
                raise NotImplementedError()
            elif state == 'reset':
                self.init_state(param)
            else:
                raise ValueError('Unsupported state policy: {}'.format(state))

    return lookahead_update_core


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

    def create_update_rule(self):
        rule = self._base_optimizer.create_update_rule()

        # Patch base update rule.
        rule.hyperparam.lookahead_lr = self.hyperparam.lr
        rule.hyperparam.lookahead_n_updates = self.hyperparam.n_updates
        rule.hyperparam.lookahead_state = self.hyperparam.state
        rule.init_state = types.MethodType(
            _to_lookahead_init_state(rule.init_state), rule)
        rule.update_core = types.MethodType(
            _to_lookahead_update_core(rule.update_core), rule)

        return rule
