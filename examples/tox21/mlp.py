from chainer import links as L
from chainer import functions as F


class MLP(chainer.Chain):

    def __init__(self, out_size):
        chain = {
            'l1' : L.Linear(None, 1024),
            'l2' : L.Linear(1024, 128),
            'l3' : L.Linear(128, 128)}
        super(MLP, self).__init__(**chain)

    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(x))
        return F.relu(self.l3(x))
