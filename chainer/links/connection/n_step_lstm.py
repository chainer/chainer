import numpy

from chainer.functions.array import reshape
from chainer.functions.connection import n_step_lstm
from chainer import link
from chainer import variable


class NStepLSTM(link.Chain):

    def __init__(self, size, n_layers):
        super(NStepLSTM, self).__init__()
        self.size = size
        self.n_layers = n_layers

        xp = self.xp

        wscale = 1

        initialW = numpy.random.normal(
            0, wscale * numpy.sqrt(1. / size), (size, size))
        self.w = variable.Variable(
            xp.empty((n_layers, 8, size, size), dtype=numpy.float32),
            volatile='auto')
        self.w.data[...] = initialW
        self.b = variable.Variable(
            xp.zeros((n_layers, 8, size)),
            volatile='auto'
        )
        self.reset_state()

    def to_cpu(self):
        super(NStepLSTM, self).to_cpu()
        self.w.to_cpu()
        self.b.to_cpu()
        if self.c is not None:
            self.c.to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super(NStepLSTM, self).to_gpu(device)
        self.w.to_gpu(device)
        self.b.to_gpu(device)
        if self.c is not None:
            self.c.to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)

    def reset_state(self):
        self.c = self.h = None

    def __call__(self, x, train=True):
        if self.c is None:
            self.c = variable.Variable(
                self.xp.zeros((self.n_layers, len(x.data), self.size), dtype=x.data.dtype),
                volatile="auto")
        if self.h is None:
            self.h = variable.Variable(
                self.xp.zeros((self.n_layers, len(x.data), self.size), dtype=x.data.dtype),
                volatile="auto")

        x = reshape.reshape(x, (1,) + x.data.shape)
        self.h, self.c, y = n_step_lstm.n_step_lstm(
            self.n_layers, self.h, self.c, x, self.w, self.b, train=train)
        y = reshape.reshape(y, y.data.shape[1:])
        return y
