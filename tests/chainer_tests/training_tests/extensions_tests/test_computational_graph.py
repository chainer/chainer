import os
import shutil
import tempfile
import unittest

import numpy

import chainer
from chainer import configuration
from chainer import links
from chainer import testing
from chainer import training
from chainer.training.extensions import computational_graph as c


class Function1(chainer.FunctionNode):

    def forward(self, inputs):
        return inputs[0],


class Function2(chainer.FunctionNode):

    def forward(self, inputs):
        return inputs[0],


class Dataset(chainer.dataset.DatasetMixin):
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def get_example(self, i):
        return numpy.array([self.values[i]], numpy.float32), numpy.int32(i % 2)


class Model(chainer.Link):
    def __init__(self):
        super(Model, self).__init__()
        self.flag_history = []
        self.l1 = links.Linear(2)
        self.l2 = links.Linear(2)
        self.i = 0

    def forward(self, x):
        self.flag_history.append(
            configuration.config.keep_graph_on_report)

        h = self.l1(x)
        if self.i == 0:
            h, = Function1().apply((h,))
        else:
            h, = Function2().apply((h,))
        h = self.l2(h)

        self.i += 1
        return h


class TestGraphBuilderKeepGraphOnReport(unittest.TestCase):

    def _run_test(self, tempdir, initial_flag):
        n_data = 4
        n_epochs = 3
        outdir = os.path.join(tempdir, 'testresult')

        # Prepare
        model = Model()
        classifier = links.Classifier(model)
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(classifier)

        dataset = Dataset([i for i in range(n_data)])
        iterator = chainer.iterators.SerialIterator(dataset, 1, shuffle=False)
        updater = training.updaters.StandardUpdater(iterator, optimizer)
        trainer = training.Trainer(updater, (n_epochs, 'epoch'), out=outdir)

        extension = c.DumpGraph('main/loss', filename='test.dot')
        trainer.extend(extension)

        # Run
        with chainer.using_config('keep_graph_on_report', initial_flag):
            trainer.run()

        # Check flag history
        self.assertEqual(model.flag_history,
                         [True] + [initial_flag] * (n_data * n_epochs - 1))

        # Check the dumped graph
        graph_path = os.path.join(outdir, 'test.dot')
        with open(graph_path) as f:
            graph_dot = f.read()

        # Check that only the first iteration is dumped
        self.assertIn('Function1', graph_dot)
        self.assertNotIn('Function2', graph_dot)

        if c.is_graphviz_available():
            self.assertTrue(os.path.exists(os.path.join(outdir, 'test.png')))

    def _check(self, initial_flag):
        tempdir = tempfile.mkdtemp()
        try:
            self._run_test(tempdir, initial_flag)
        finally:
            shutil.rmtree(tempdir)

    def test_keep_graph_on_report_flag_true(self):
        self._check(True)

    def test_keep_graph_on_report_flag_false(self):
        self._check(False)


testing.run_module(__name__, __file__)
