import chainer
from chainer import functions as F
import six

from mlp import MLP


class Model(chainer.Chain):

    def __init__(self, unit_num, task_num):
        branch = chainer.ChainList(*[
            MLP(unit_num, 1, False) for _ in six.moves.range(task_num)])
        super(Model, self).__init__(
            base=MLP(unit_num, unit_num),
            branch=branch)
        self.train = True

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, val):
        self._train = val
        self.base.train = val
        for l in self.branch:
            l.train = val

    def __call__(self, x):
        x = self.base(x)
        xs = [f(x) for f in self.branch]
        y = F.concat(xs, axis=1)
        return y
