import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

from benchmarks import BenchmarkBase
from benchmarks.utils import backends
from benchmarks.utils import is_backend_gpu
from benchmarks.utils import is_backend_ideep
from benchmarks.utils import parameterize


class Network(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(Network, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_units)
            self.l2 = L.Linear(None, n_units)
            self.l3 = L.Linear(None, n_out)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


class Application(object):

    def main(self, units, epoch, batchsize):
        model = L.Classifier(Network(units, 10))

        gpu = -1
        if is_backend_gpu():
            gpu = 0
            chainer.cuda.get_device_from_id(gpu).use()
            model.to_gpu()
        elif is_backend_ideep():
            model.to_intel64()

        optimizer = chainer.optimizers.MomentumSGD()
        optimizer.setup(model)

        train, test = chainer.datasets.get_mnist()
        train_iter = chainer.iterators.SerialIterator(
            train, batchsize)
        test_iter = chainer.iterators.SerialIterator(
            test, batchsize, repeat=False, shuffle=False)

        updater = training.updater.StandardUpdater(
            train_iter, optimizer, device=gpu)
        trainer = training.Trainer(updater, (epoch, 'epoch'))
        trainer.extend(extensions.Evaluator(test_iter, model, device=gpu))

        trainer.run()


@backends('gpu', 'gpu-cudnn', 'cpu', 'cpu-ideep')
@parameterize([
    ('units', [10, 100, 150]),
])
class MLP(BenchmarkBase):
    timeout = 360

    def time_overall(self, units):
        Application().main(units=units, epoch=1, batchsize=100)
