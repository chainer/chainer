import unittest

import numpy

import chainer
from chainer import dataset
from chainer import testing
from chainer.training import extensions


class DummyModel(chainer.Chain):

    def __init__(self, test):
        super(DummyModel, self).__init__()
        self.args = []
        self.test = test

    def __call__(self, x):
        self.args.append(x)
        chainer.report({'loss': x.sum()}, self)


class DummyModelTwoArgs(chainer.Chain):

    def __init__(self, test):
        super(DummyModelTwoArgs, self).__init__()
        self.args = []
        self.test = test

    def __call__(self, x, y):
        self.args.append((x, y))
        chainer.report({'loss': x.sum() + y.sum()}, self)


class DummyIterator(dataset.Iterator):

    def __init__(self, return_values):
        self.iterator = iter(return_values)
        self.finalized = False

    def __next__(self):
        return next(self.iterator)

    def finalize(self):
        self.finalized = True


class DummyConverter(object):

    def __init__(self, return_values):
        self.args = []
        self.iterator = iter(return_values)

    def __call__(self, batch, device):
        self.args.append({'batch': batch, 'device': device})
        return next(self.iterator)


class TestEvaluator(unittest.TestCase):

    def setUp(self):
        self.data = [
            numpy.random.uniform(-1, 1, (3, 4)).astype('f') for _ in range(2)]
        self.batches = [
            numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f')
            for _ in range(2)]

        self.iterator = DummyIterator(self.data)
        self.converter = DummyConverter(self.batches)
        self.target = DummyModel(self)
        self.evaluator = extensions.Evaluator(
            self.iterator, self.target, converter=self.converter)
        self.expect_mean = numpy.mean([numpy.sum(x) for x in self.batches])

    def test_evaluate(self):
        reporter = chainer.Reporter()
        reporter.add_observer('target', self.target)
        with reporter:
            mean = self.evaluator.evaluate()

        # No observation is reported to the current reporter. Instead the
        # evaluator collect results in order to calculate their mean.
        self.assertEqual(len(reporter.observation), 0)

        # The converter gets results of the iterator.
        self.assertEqual(len(self.converter.args), len(self.data))
        for i in range(len(self.data)):
            numpy.testing.assert_array_equal(
                self.converter.args[i]['batch'], self.data[i])
            self.assertIsNone(self.converter.args[i]['device'])

        # The model gets results of converter.
        self.assertEqual(len(self.target.args), len(self.batches))
        for i in range(len(self.batches)):
            numpy.testing.assert_array_equal(
                self.target.args[i], self.batches[i])

        self.assertAlmostEqual(mean['target/loss'], self.expect_mean, places=4)

        self.evaluator.finalize()
        self.assertTrue(self.iterator.finalized)

    def test_call(self):
        mean = self.evaluator()
        # 'main' is used by default
        self.assertAlmostEqual(mean['main/loss'], self.expect_mean, places=4)

    def test_evaluator_name(self):
        self.evaluator.name = 'eval'
        mean = self.evaluator()
        # name is used as a prefix
        self.assertAlmostEqual(
            mean['eval/main/loss'], self.expect_mean, places=4)

    def test_current_report(self):
        reporter = chainer.Reporter()
        with reporter:
            mean = self.evaluator()
        # The result is reported to the current reporter.
        self.assertEqual(reporter.observation, mean)


class TestEvaluatorTupleData(unittest.TestCase):

    def setUp(self):
        self.data = [
            numpy.random.uniform(-1, 1, (3, 4)).astype('f') for _ in range(2)]
        self.batches = [
            (numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f'),
             numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f'))
            for _ in range(2)]

        self.iterator = DummyIterator(self.data)
        self.converter = DummyConverter(self.batches)
        self.target = DummyModelTwoArgs(self)
        self.evaluator = extensions.Evaluator(
            self.iterator, self.target, converter=self.converter, device=1)

    def test_evaluate(self):
        reporter = chainer.Reporter()
        reporter.add_observer('target', self.target)
        with reporter:
            mean = self.evaluator.evaluate()

        # The converter gets results of the iterator and the device number.
        self.assertEqual(len(self.converter.args), len(self.data))
        for i in range(len(self.data)):
            numpy.testing.assert_array_equal(
                self.converter.args[i]['batch'], self.data[i])
            self.assertEqual(self.converter.args[i]['device'], 1)

        # The model gets results of converter.
        self.assertEqual(len(self.target.args), len(self.batches))
        for i in range(len(self.batches)):
            numpy.testing.assert_array_equal(
                self.target.args[i], self.batches[i])

        expect_mean = numpy.mean([numpy.sum(x) for x in self.batches])
        self.assertAlmostEqual(mean['target/loss'], expect_mean, places=4)


class TestEvaluatorDictData(unittest.TestCase):

    def setUp(self):
        self.data = range(2)
        self.batches = [
            {'x': numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f'),
             'y': numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f')}
            for _ in range(2)]

        self.iterator = DummyIterator(self.data)
        self.converter = DummyConverter(self.batches)
        self.target = DummyModelTwoArgs(self)
        self.evaluator = extensions.Evaluator(
            self.iterator, self.target, converter=self.converter)

    def test_evaluate(self):
        reporter = chainer.Reporter()
        reporter.add_observer('target', self.target)
        with reporter:
            mean = self.evaluator.evaluate()

        # The model gets results of converter.
        self.assertEqual(len(self.target.args), len(self.batches))
        for i in range(len(self.batches)):
            numpy.testing.assert_array_equal(
                self.target.args[i][0], self.batches[i]['x'])
            numpy.testing.assert_array_equal(
                self.target.args[i][1], self.batches[i]['y'])

        expect_mean = numpy.mean(
            [numpy.sum(x['x']) + numpy.sum(x['y']) for x in self.batches])
        self.assertAlmostEqual(mean['target/loss'], expect_mean, places=4)


class TestEvaluatorWithEvalFunc(unittest.TestCase):

    def setUp(self):
        self.data = [
            numpy.random.uniform(-1, 1, (3, 4)).astype('f') for _ in range(2)]
        self.batches = [
            numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f')
            for _ in range(2)]

        self.iterator = DummyIterator(self.data)
        self.converter = DummyConverter(self.batches)
        self.target = DummyModel(self)
        self.evaluator = extensions.Evaluator(
            self.iterator, {}, converter=self.converter,
            eval_func=self.target)

    def test_evaluate(self):
        reporter = chainer.Reporter()
        reporter.add_observer('target', self.target)
        with reporter:
            self.evaluator.evaluate()

        # The model gets results of converter.
        self.assertEqual(len(self.target.args), len(self.batches))
        for i in range(len(self.batches)):
            numpy.testing.assert_array_equal(
                self.target.args[i], self.batches[i])


testing.run_module(__name__, __file__)
