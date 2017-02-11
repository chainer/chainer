import chainer
from chainer import functions as F
from mlp import MLP


class Model(chainer.Chain):

    def __init__(self, task_num):
        branch = chainer.ChainList([MLP(1) for _ in task_num])
        super(Model, self).__init__(
            base=MLP(128),
            branch=branch)

    def __call__(self, x):
        x = self.base(x)
        xs = [f(x) for f in self.branch]
        return F.concatenate(xs)
