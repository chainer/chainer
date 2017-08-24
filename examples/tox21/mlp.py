import chainer
from chainer import functions as F
from chainer import links as L


class MLP(chainer.Chain):

    def __init__(self, unit_num, out_num, last_relu=True):
        super(MLP, self).__init__(
            l1=L.Linear(None, unit_num),
            l2=L.Linear(unit_num, out_num))
        self.last_relu = last_relu
        self.train = True

    def __call__(self, x):
        h = F.relu(F.dropout(self.l1(x), train=self.train))
        h = self.l2(h)
        if self.last_relu:
            h = F.relu(F.dropout(h, train=self.train))
        return h
