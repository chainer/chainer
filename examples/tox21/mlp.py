import chainer
from chainer import functions as F
from chainer import links as L


class MLP(chainer.Chain):

    def __init__(self, unit_num, out_num, last_relu=True):
        chain = {
            'l1': L.Linear(None, unit_num),
            'l2': L.Linear(unit_num, out_num)}
        super(MLP, self).__init__(**chain)
        self.last_relu = last_relu

    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = self.l2(h)
        if self.last_relu:
            h = F.relu(h)
        return h
