import chainer
from chainer import functions as F
import six

from mlp import MLP


class Model(chainer.Chain):

    def __init__(self, task_num):
        branch = chainer.ChainList(*[
            MLP(1) for _ in six.moves.range(task_num)])
        super(Model, self).__init__(
            base=MLP(256),
            branch=branch)

    def __call__(self, x):
        x = self.base(x)
        xs = [f(x) for f in self.branch]
        y = F.concat(xs, axis=1)
        print(y[0].data)
        return y
