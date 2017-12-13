import contextlib
import tempfile
import unittest

import numpy

import chainer
from chainer import configuration
from chainer import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr


class TestReporter(unittest.TestCase):

    def test_empty_reporter(self):
        reporter = chainer.Reporter()
        self.assertEqual(reporter.observation, {})

    def test_enter_exit(self):
        reporter1 = chainer.Reporter()
        reporter2 = chainer.Reporter()
        with reporter1:
            self.assertIs(chainer.get_current_reporter(), reporter1)
            with reporter2:
                self.assertIs(chainer.get_current_reporter(), reporter2)
            self.assertIs(chainer.get_current_reporter(), reporter1)

    def test_scope(self):
        reporter1 = chainer.Reporter()
        reporter2 = chainer.Reporter()
        with reporter1:
            observation = {}
            with reporter2.scope(observation):
                self.assertIs(chainer.get_current_reporter(), reporter2)
                self.assertIs(reporter2.observation, observation)
            self.assertIs(chainer.get_current_reporter(), reporter1)
            self.assertIsNot(reporter2.observation, observation)

    def test_add_observer(self):
        reporter = chainer.Reporter()
        observer = object()
        reporter.add_observer('o', observer)

        reporter.report({'x': 1}, observer)

        observation = reporter.observation
        self.assertIn('o/x', observation)
        self.assertEqual(observation['o/x'], 1)
        self.assertNotIn('x', observation)

    def test_add_observers(self):
        reporter = chainer.Reporter()
        observer1 = object()
        reporter.add_observer('o1', observer1)
        observer2 = object()
        reporter.add_observer('o2', observer2)

        reporter.report({'x': 1}, observer1)
        reporter.report({'y': 2}, observer2)

        observation = reporter.observation
        self.assertIn('o1/x', observation)
        self.assertEqual(observation['o1/x'], 1)
        self.assertIn('o2/y', observation)
        self.assertEqual(observation['o2/y'], 2)
        self.assertNotIn('x', observation)
        self.assertNotIn('y', observation)
        self.assertNotIn('o1/y', observation)
        self.assertNotIn('o2/x', observation)

    def test_report_without_observer(self):
        reporter = chainer.Reporter()
        reporter.report({'x': 1})

        observation = reporter.observation
        self.assertIn('x', observation)
        self.assertEqual(observation['x'], 1)


class TestKeepGraphOnReportFlag(unittest.TestCase):

    @contextlib.contextmanager
    def _scope(self, flag):
        # If flag is None, return the nop context.
        # Otherwise, return the context in which
        # keep_graph_on_report is set temporarily.
        old = configuration.config.keep_graph_on_report
        if flag is not None:
            configuration.config.keep_graph_on_report = flag
        try:
            yield
        finally:
            configuration.config.keep_graph_on_report = old

    def test_keep_graph_default(self):
        x = chainer.Variable(numpy.array([1], numpy.float32))
        y, = functions.Sigmoid().apply((x,))
        reporter = chainer.Reporter()
        with self._scope(None):
            reporter.report({'y': y})
        self.assertIsNone(reporter.observation['y'].creator)

    def test_keep_graph(self):
        x = chainer.Variable(numpy.array([1], numpy.float32))
        y, = functions.Sigmoid().apply((x,))
        reporter = chainer.Reporter()
        with self._scope(True):
            reporter.report({'y': y})
        self.assertIsInstance(reporter.observation['y'].creator,
                              functions.Sigmoid)

    def test_not_keep_graph(self):
        x = chainer.Variable(numpy.array([1], numpy.float32))
        y, = functions.Sigmoid().apply((x,))
        reporter = chainer.Reporter()
        with self._scope(False):
            reporter.report({'y': y})
        self.assertIsNone(reporter.observation['y'].creator)


class TestReport(unittest.TestCase):

    def test_report_without_reporter(self):
        observer = object()
        chainer.report({'x': 1}, observer)

    def test_report(self):
        reporter = chainer.Reporter()
        with reporter:
            chainer.report({'x': 1})
        observation = reporter.observation
        self.assertIn('x', observation)
        self.assertEqual(observation['x'], 1)

    def test_report_with_observer(self):
        reporter = chainer.Reporter()
        observer = object()
        reporter.add_observer('o', observer)
        with reporter:
            chainer.report({'x': 1}, observer)
        observation = reporter.observation
        self.assertIn('o/x', observation)
        self.assertEqual(observation['o/x'], 1)

    def test_report_with_unregistered_observer(self):
        reporter = chainer.Reporter()
        observer = object()
        with reporter:
            with self.assertRaises(KeyError):
                chainer.report({'x': 1}, observer)

    def test_report_scope(self):
        reporter = chainer.Reporter()
        observation = {}

        with reporter:
            with chainer.report_scope(observation):
                chainer.report({'x': 1})

        self.assertIn('x', observation)
        self.assertEqual(observation['x'], 1)
        self.assertNotIn('x', reporter.observation)


class TestSummary(unittest.TestCase):

    def setUp(self):
        self.summary = chainer.reporter.Summary()

    def test_numpy(self):
        self.summary.add(numpy.array(1, 'f'))
        self.summary.add(numpy.array(-2, 'f'))

        mean = self.summary.compute_mean()
        testing.assert_allclose(mean, numpy.array(-0.5, 'f'))

        mean, std = self.summary.make_statistics()
        testing.assert_allclose(mean, numpy.array(-0.5, 'f'))
        testing.assert_allclose(std, numpy.array(1.5, 'f'))

    @attr.gpu
    def test_cupy(self):
        xp = cuda.cupy
        self.summary.add(xp.array(1, 'f'))
        self.summary.add(xp.array(-2, 'f'))

        mean = self.summary.compute_mean()
        testing.assert_allclose(mean, numpy.array(-0.5, 'f'))

        mean, std = self.summary.make_statistics()
        testing.assert_allclose(mean, numpy.array(-0.5, 'f'))
        testing.assert_allclose(std, numpy.array(1.5, 'f'))

    def test_int(self):
        self.summary.add(1)
        self.summary.add(2)
        self.summary.add(3)

        mean = self.summary.compute_mean()
        testing.assert_allclose(mean, 2)

        mean, std = self.summary.make_statistics()
        testing.assert_allclose(mean, 2)
        testing.assert_allclose(std, numpy.sqrt(2 / 3))

    def test_float(self):
        self.summary.add(1.)
        self.summary.add(2.)
        self.summary.add(3.)

        mean = self.summary.compute_mean()
        testing.assert_allclose(mean, 2.)

        mean, std = self.summary.make_statistics()
        testing.assert_allclose(mean, 2.)
        testing.assert_allclose(std, numpy.sqrt(2. / 3.))

    def test_serialize(self):
        self.summary.add(1.)
        self.summary.add(2.)

        summary = chainer.reporter.Summary()
        testing.save_and_load_npz(self.summary, summary)
        summary.add(3.)

        mean = summary.compute_mean()
        testing.assert_allclose(mean, 2.)

        mean, std = summary.make_statistics()
        testing.assert_allclose(mean, 2.)
        testing.assert_allclose(std, numpy.sqrt(2. / 3.))

    def test_serialize_backward_compat(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            # old version does not save anything
            numpy.savez(f, dummy=0)
            chainer.serializers.load_npz(f.name, self.summary)

        self.summary.add(2.)
        self.summary.add(3.)

        mean = self.summary.compute_mean()
        testing.assert_allclose(mean, 2.5)

        mean, std = self.summary.make_statistics()
        testing.assert_allclose(mean, 2.5)
        testing.assert_allclose(std, 0.5)


class TestDictSummary(unittest.TestCase):

    def setUp(self):
        self.summary = chainer.reporter.DictSummary()

    def test(self):
        self.summary.add({'numpy': numpy.array(3, 'f'), 'int': 1, 'float': 4.})
        self.summary.add({'numpy': numpy.array(1, 'f'), 'int': 5, 'float': 9.})
        self.summary.add({'numpy': numpy.array(2, 'f'), 'int': 6, 'float': 5.})
        self.summary.add({'numpy': numpy.array(3, 'f'), 'int': 5, 'float': 8.})

        mean = self.summary.compute_mean()
        self.assertEqual(set(mean.keys()), {'numpy', 'int', 'float'})
        testing.assert_allclose(mean['numpy'], 9. / 4.)
        testing.assert_allclose(mean['int'], 17 / 4)
        testing.assert_allclose(mean['float'], 13. / 2.)

        stats = self.summary.make_statistics()
        self.assertEqual(
            set(stats.keys()),
            {'numpy',  'int',  'float', 'numpy.std', 'int.std', 'float.std'})
        testing.assert_allclose(stats['numpy'], 9. / 4.)
        testing.assert_allclose(stats['int'], 17 / 4)
        testing.assert_allclose(stats['float'], 13. / 2.)
        testing.assert_allclose(stats['numpy.std'], numpy.sqrt(11. / 16.))
        testing.assert_allclose(stats['int.std'], numpy.sqrt(59 / 16))
        testing.assert_allclose(stats['float.std'], numpy.sqrt(17. / 4.))

    @attr.gpu
    def test_cupy(self):
        xp = cuda.cupy
        self.summary.add({'cupy': xp.array(3, 'f'), 'int': 1, 'float': 4.})
        self.summary.add({'cupy': xp.array(1, 'f'), 'int': 5, 'float': 9.})
        self.summary.add({'cupy': xp.array(2, 'f'), 'int': 6, 'float': 5.})
        self.summary.add({'cupy': xp.array(3, 'f'), 'int': 5, 'float': 8.})

        mean = self.summary.compute_mean()
        self.assertEqual(set(mean.keys()), {'cupy', 'int', 'float'})
        testing.assert_allclose(mean['cupy'], 9. / 4.)
        testing.assert_allclose(mean['int'], 17 / 4)
        testing.assert_allclose(mean['float'], 13. / 2.)

        stats = self.summary.make_statistics()
        self.assertEqual(
            set(stats.keys()),
            {'cupy',  'int',  'float', 'cupy.std', 'int.std', 'float.std'})
        testing.assert_allclose(stats['cupy'], 9. / 4.)
        testing.assert_allclose(stats['int'], 17 / 4)
        testing.assert_allclose(stats['float'], 13. / 2.)
        testing.assert_allclose(stats['cupy.std'], numpy.sqrt(11.) / 4.)
        testing.assert_allclose(stats['int.std'], numpy.sqrt(59) / 4)
        testing.assert_allclose(stats['float.std'], numpy.sqrt(17.) / 2.)

    def test_sparse(self):
        self.summary.add({'numpy': numpy.array(3, 'f'), 'int': 1})
        self.summary.add({'numpy': numpy.array(1, 'f'), 'int': 5, 'float': 9.})
        self.summary.add({'int': 6})
        self.summary.add({'numpy': numpy.array(3, 'f'), 'int': 5, 'float': 8.})

        mean = self.summary.compute_mean()
        self.assertEqual(set(mean.keys()), {'numpy', 'int', 'float'})
        testing.assert_allclose(mean['numpy'], 7. / 3.)
        testing.assert_allclose(mean['int'], 17 / 4)
        testing.assert_allclose(mean['float'], 17. / 2.)

        stats = self.summary.make_statistics()
        self.assertEqual(
            set(stats.keys()),
            {'numpy',  'int',  'float', 'numpy.std', 'int.std', 'float.std'})
        testing.assert_allclose(stats['numpy'], 7. / 3.)
        testing.assert_allclose(stats['int'], 17 / 4)
        testing.assert_allclose(stats['float'], 17. / 2.)
        testing.assert_allclose(stats['numpy.std'], numpy.sqrt(8. / 9.))
        testing.assert_allclose(stats['int.std'], numpy.sqrt(59 / 16))
        testing.assert_allclose(stats['float.std'], 1. / 2.)

    def test_serialize(self):
        self.summary.add({'numpy': numpy.array(3, 'f'), 'int': 1, 'float': 4.})
        self.summary.add({'numpy': numpy.array(1, 'f'), 'int': 5, 'float': 9.})
        self.summary.add({'numpy': numpy.array(2, 'f'), 'int': 6, 'float': 5.})

        summary = chainer.reporter.DictSummary()
        testing.save_and_load_npz(self.summary, summary)
        summary.add({'numpy': numpy.array(3, 'f'), 'int': 5, 'float': 8.})

        mean = summary.compute_mean()
        self.assertEqual(set(mean.keys()), {'numpy', 'int', 'float'})
        testing.assert_allclose(mean['numpy'], 9. / 4.)
        testing.assert_allclose(mean['int'], 17 / 4)
        testing.assert_allclose(mean['float'], 13. / 2.)

        stats = summary.make_statistics()
        self.assertEqual(
            set(stats.keys()),
            {'numpy',  'int',  'float', 'numpy.std', 'int.std', 'float.std'})
        testing.assert_allclose(stats['numpy'], 9. / 4.)
        testing.assert_allclose(stats['int'], 17 / 4)
        testing.assert_allclose(stats['float'], 13. / 2.)
        testing.assert_allclose(stats['numpy.std'], numpy.sqrt(11. / 16.))
        testing.assert_allclose(stats['int.std'], numpy.sqrt(59 / 16))
        testing.assert_allclose(stats['float.std'], numpy.sqrt(17. / 4.))

    def test_serialize_names_with_slash(self):
        self.summary.add({'a/b': 3., '/a/b': 1., 'a/b/': 4.})
        self.summary.add({'a/b': 1., '/a/b': 5., 'a/b/': 9.})
        self.summary.add({'a/b': 2., '/a/b': 6., 'a/b/': 5.})

        summary = chainer.reporter.DictSummary()
        testing.save_and_load_npz(self.summary, summary)
        summary.add({'a/b': 3., '/a/b': 5., 'a/b/': 8.})

        mean = summary.compute_mean()
        self.assertEqual(set(mean.keys()), {'a/b', '/a/b', 'a/b/'})
        testing.assert_allclose(mean['a/b'], 9. / 4.)
        testing.assert_allclose(mean['/a/b'], 17 / 4)
        testing.assert_allclose(mean['a/b/'], 13. / 2.)

        stats = summary.make_statistics()
        self.assertEqual(
            set(stats.keys()),
            {'a/b',  '/a/b',  'a/b/', 'a/b.std', '/a/b.std', 'a/b/.std'})
        testing.assert_allclose(stats['a/b'], 9. / 4.)
        testing.assert_allclose(stats['/a/b'], 17 / 4)
        testing.assert_allclose(stats['a/b/'], 13. / 2.)
        testing.assert_allclose(stats['a/b.std'], numpy.sqrt(11. / 16.))
        testing.assert_allclose(stats['/a/b.std'], numpy.sqrt(59 / 16))
        testing.assert_allclose(stats['a/b/.std'], numpy.sqrt(17. / 4.))

    def test_serialize_backward_compat(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            # old version does not save anything
            numpy.savez(f, dummy=0)
            chainer.serializers.load_npz(f.name, self.summary)

        self.summary.add({'numpy': numpy.array(3, 'f'), 'int': 1, 'float': 4.})
        self.summary.add({'numpy': numpy.array(1, 'f'), 'int': 5, 'float': 9.})
        self.summary.add({'numpy': numpy.array(2, 'f'), 'int': 6, 'float': 5.})
        self.summary.add({'numpy': numpy.array(3, 'f'), 'int': 5, 'float': 8.})

        mean = self.summary.compute_mean()
        self.assertEqual(set(mean.keys()), {'numpy', 'int', 'float'})
        testing.assert_allclose(mean['numpy'], 9. / 4.)
        testing.assert_allclose(mean['int'], 17 / 4)
        testing.assert_allclose(mean['float'], 13. / 2.)

        stats = self.summary.make_statistics()
        self.assertEqual(
            set(stats.keys()),
            {'numpy',  'int',  'float', 'numpy.std', 'int.std', 'float.std'})
        testing.assert_allclose(stats['numpy'], 9. / 4.)
        testing.assert_allclose(stats['int'], 17 / 4)
        testing.assert_allclose(stats['float'], 13. / 2.)
        testing.assert_allclose(stats['numpy.std'], numpy.sqrt(11. / 16.))
        testing.assert_allclose(stats['int.std'], numpy.sqrt(59 / 16))
        testing.assert_allclose(stats['float.std'], numpy.sqrt(17. / 4.))


testing.run_module(__name__, __file__)
