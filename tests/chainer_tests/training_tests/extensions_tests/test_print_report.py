import sys
import unittest

from chainer import testing
from chainer.training import extensions


class TestPrintReport(unittest.TestCase):

    def test_stream_typecheck(self):
        report = extensions.PrintReport(['epoch'], out=sys.stderr)
        self.assertIsInstance(report, extensions.PrintReport)

        with self.assertRaises(TypeError):
            report = extensions.PrintReport(['epoch'], out=False)


testing.run_module(__name__, __file__)
