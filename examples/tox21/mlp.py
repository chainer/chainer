import chainer
from chainer import functions as F
from chainer import links as L


class MLP(chainer.Chain):

    def __init__(self, out_size, last_relu=True):
        chain = {
            'l1': L.Linear(None, 1024),
            'l2': L.Linear(1024, out_size)}
        super(MLP, self).__init__(**chain)
        self.last_relu = last_relu

    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = self.l2(h)
        if self.last_relu:
            h = F.relu(h)
        return h
