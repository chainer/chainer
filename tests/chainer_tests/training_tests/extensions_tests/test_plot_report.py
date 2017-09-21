import unittest
import warnings

from chainer import testing
from chainer.training import extensions


class TestPlotReport(unittest.TestCase):

    def test_available(self):
        try:
            import matplotlib  # NOQA
            available = True
        except ImportError:
            available = False

        with warnings.catch_warnings(record=True) as w:
            self.assertEqual(extensions.PlotReport.available(), available)

        # It shows warning only when matplotlib is not available
        if available:
            self.assertEqual(len(w), 0)
        else:
            self.assertEqual(len(w), 1)

    # In the following we explicitly use plot_report._available instead of
    # PlotReport.available() because in some cases `test_available()` fails
    # because it sometimes does not raise UserWarning despite
    # matplotlib is not installed (this is due to the difference between
    # the behavior of unittest in python2 and that in python3).
    @unittest.skipUnless(
        extensions.plot_report._available, 'matplotlib is not installed')
    def test_lazy_import(self):
        # To support python2, we do not use self.assertWarns()
        with warnings.catch_warnings(record=True) as w:
            import matplotlib
            matplotlib.use('Agg')

        self.assertEqual(len(w), 0)


testing.run_module(__name__, __file__)
