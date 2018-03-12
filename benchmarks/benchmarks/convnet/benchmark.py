import chainer
from chainer import cuda
from chainer import optimizers

import cupy
import numpy

from benchmarks import BenchmarkBase
from benchmarks.utils import backends
from benchmarks.utils import is_backend_gpu
from benchmarks.utils import is_backend_ideep
from benchmarks.utils import parameterize


class _ConvnetBase(BenchmarkBase):
    """
    Benchmark code from convnet-benchmark.

    https://github.com/soumith/convnet-benchmarks/tree/master/chainer
    """

    timeout = 600
    number = 1

    def setup(self, arch, batchsize):
        xp = self.xp

        if arch == 'alexnet':
            from benchmarks.convnet.nets import alex
            model = alex.Alex()
        elif arch == 'googlenet':
            from benchmarks.convnet.nets import googlenet
            model = googlenet.GoogLeNet()
        elif arch == 'vgga':
            from benchmarks.convnet.nets import vgga
            model = vgga.vgga()
        elif arch == 'overfeat':
            from benchmarks.convnet.nets import overfeat
            model = overfeat.overfeat()
        else:
            raise ValueError('Invalid architecture name')

        if is_backend_gpu():
            model.to_gpu()
        elif is_backend_ideep():
            model.to_intel64()

        # Setup optimizer
        optimizer = optimizers.SGD(lr=0.01)
        optimizer.setup(model)

        # Set cuDNN workspace size
        workspace_size = int(1 * 2**30)
        chainer.cuda.set_max_workspace_size(workspace_size)

        chainer.config.train = True

        x = xp.ndarray((batchsize, 3, model.insize,
                           model.insize), dtype=xp.float32)
        x.fill(33333)

        if arch == 'googlenet':
            out1, out2, out3 = model.forward(x)
            out = out1 + out2 + out3
        else:
            out = model.forward(x)

        out.zerograd()
        out.grad.fill(3)
        model.cleargrads()

        self._x = x
        self._model = model
        self._out = out

    def time_forward(self, arch, batchsize):
        self._model.forward(self._x)

    def time_backward(self, arch, batchsize):
        self._out.backward()


@backends('gpu', 'gpu-cudnn')
@parameterize([
    ('arch', ['vgga']),
    ('batchsize', [32]),
])
class ConvnetVGGA(_ConvnetBase):
    pass


@backends('gpu', 'gpu-cudnn')
@parameterize([
    ('arch', ['alexnet', 'googlenet', 'overfeat']),
    ('batchsize', [128]),
])
class ConvnetOthers(_ConvnetBase):
    pass
