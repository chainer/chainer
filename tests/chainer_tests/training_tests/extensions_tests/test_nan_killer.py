import os
import tempfile
import unittest

import numpy

import chainer
from chainer import links
from chainer import testing
from chainer.testing import attr
from chainer import training


class Model(chainer.Chain):

    def __init__(self):
        super(Model, self).__init__()
        with self.init_scope():
            self.l = links.Linear(1, 3)

    def __call__(self, x):
        return self.l(x)


class Dataset(chainer.dataset.DatasetMixin):

    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def get_example(self, i):
        return numpy.array([self.values[i]], numpy.float32), numpy.int32(i % 2)


class TestNaNKiller(unittest.TestCase):

    def setUp(self):
        self.n_data = 4
        self.n_epochs = 3

        self.model = Model()
        self.classifier = links.Classifier(self.model)
        self.optimizer = chainer.optimizers.Adam()
        self.optimizer.setup(self.classifier)

        self.dataset = Dataset([i for i in range(self.n_data)])
        self.iterator = chainer.iterators.SerialIterator(
            self.dataset, 1, shuffle=False)

    def prepare(self, device=None):
        tempdir = tempfile.mkdtemp()
        outdir = os.path.join(tempdir, 'testresult')
        self.updater = training.updaters.StandardUpdater(
            self.iterator, self.optimizer, device=device)
        self.trainer = training.Trainer(
            self.updater, (self.n_epochs, 'epoch'), out=outdir)
        self.trainer.extend(training.extensions.NaNKiller())

    def test_trainer(self):
        self.prepare()
        self.trainer.run()

    def test_nan_killer(self):
        self.prepare()
        self.model.l.W.array[1, 0] = numpy.nan
        with self.assertRaises(RuntimeError):
            self.trainer.run(show_loop_exception_msg=False)

    @attr.gpu
    def test_trainer_gpu(self):
        self.prepare(device=0)
        self.trainer.run()

    @attr.gpu
    def test_nan_killer_gpu(self):
        self.prepare(device=0)
        self.model.l.W.array[:] = numpy.nan
        with self.assertRaises(RuntimeError):
            self.trainer.run(show_loop_exception_msg=False)


testing.run_module(__name__, __file__)
