import os
import unittest


import numpy
import chainer
from chainer import testing
from chainer.training import extensions


class TestVariableStatisticsPlot(unittest.TestCase):

    def setUp(self):
        stop_trigger = (2, 'iteration')
        extension_trigger = (1, 'iteration')
        self.file_name = 'variable_statistics_plot_test.png'

        self.trainer = testing.get_trainer_with_mock_updater(
            stop_trigger=stop_trigger)

        x = numpy.random.rand(1, 2, 3)
        self.extension = extensions.VariableStatisticsPlot(
            chainer.variable.Variable(x), trigger=extension_trigger,
            file_name=self.file_name)
        self.trainer.extend(self.extension, extension_trigger)

    # In the following we explicitly use plot_report._available instead of
    # PlotReport.available() because in some cases `test_available()` fails
    # because it sometimes does not raise UserWarning despite
    # matplotlib is not installed (this is due to the difference between
    # the behavior of unittest in python2 and that in python3).
    @unittest.skipUnless(
        extensions.variable_statistics_plot._available,
        'matplotlib is not installed')
    def test_run_and_save_plot(self):
        import matplotlib
        matplotlib.use('Agg')
        try:
            self.trainer.run()
        finally:
            os.remove(os.path.join(self.trainer.out, self.file_name))


@testing.parameterize(
    {'shape': (2, 7, 3), 'n': 5, 'reservoir_size': 3}
)
class TestReservoir(unittest.TestCase):

    def setUp(self):
        self.xs = [
            numpy.random.uniform(-1, 1, self.shape) for i in range(self.n)]

    def test_reservoir_size(self):
        self.reservoir = extensions.variable_statistics_plot.Reservoir(
            size=self.reservoir_size, data_shape=self.shape)
        for x in self.xs:
            self.reservoir.add(x)
        idxs, data = self.reservoir.get_data()
        self.assertEqual(len(idxs), self.reservoir_size)
        self.assertEqual(len(data), self.reservoir_size)
        self.assertEqual(idxs.ndim, 1)
        self.assertEqual(data[0].shape, self.xs[0].shape)
        testing.assert_allclose(idxs, numpy.sort(idxs))


@testing.parameterize(
    {'shape': (2, 7, 3)}
)
class TestStatistician(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape)

    def test_statistician_percentiles(self):
        self.percentiles = (0., 50., 100.)  # min, median, max
        self.statistician = extensions.variable_statistics_plot.Statistician(
            percentiles=self.percentiles)
        stat = self.statistician(self.x, axis=None, dtype=self.x.dtype)
        self.assertEqual(stat.size, self.statistician.data_size)
        self.assertEqual(stat.dtype, self.x.dtype)
        self.assertAlmostEqual(stat[0], numpy.mean(self.x))
        self.assertAlmostEqual(stat[1], numpy.std(self.x))
        self.assertAlmostEqual(stat[2], numpy.min(self.x))
        self.assertAlmostEqual(stat[3], numpy.median(self.x))
        self.assertAlmostEqual(stat[4], numpy.max(self.x))


testing.run_module(__name__, __file__)
