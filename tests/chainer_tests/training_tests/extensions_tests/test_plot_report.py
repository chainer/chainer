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

        # It shows warning only when matplotlib.pyplot is not available
        if available:
            self.assertEqual(len(w), 0)
        else:
            self.assertEqual(len(w), 1)

    def test_lazy_import(self):
        if extensions.PlotReport.available():
            # To support python2, we do not use self.assertWarns()
            with warnings.catch_warnings(record=True) as w:
                import matplotlib
                matplotlib.use('Agg')

            self.assertEqual(len(w), 0)


testing.run_module(__name__, __file__)
