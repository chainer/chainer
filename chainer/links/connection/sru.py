from chainer import cuda
import chainer.functions as F
from chainer import link
from chainer.links.connection import linear


class SRU(link.Chain):

    def __init__(self, in_size, out_size=None):
        super(SRU, self).__init__()
        self.activation = F.tanh
        with self.init_scope():
            self.W = linear.Linear(in_size, out_size * 3)
        self.c = None

    def reset_state(self):
        self.c = None

    def __call__(self, x):
        u = self.W(x)
        if self.c is None:
            c = cuda.cupy.zeros(x.shape, 'f')
        else:
            c = self.c
        self.c, h = F.sru(c, x, u)
        return h
