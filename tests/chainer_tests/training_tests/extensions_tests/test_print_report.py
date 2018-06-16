import sys
import unittest

from mock import MagicMock

from chainer import testing
from chainer.training import extensions


class TestPrintReportInitialization(unittest.TestCase):
    def test_stream_as_out_passes(self):
        stream = MagicMock(spec=sys.stdout)
        report = extensions.PrintReport(['epoch'], out=stream)
        self.assertIsInstance(report, extensions.PrintReport)

    def test_object_without_write_as_out_does_not_pass(self):
        stream = MagicMock()
        del stream.write
        with self.assertRaises(TypeError):
            extensions.PrintReport(['epoch'], out=stream)

    def test_stream_without_flush_as_out_passes(self):
        stream = MagicMock(spec=sys.stderr)
        del stream.flush
        report = extensions.PrintReport(['epoch'], out=stream)
        self.assertIsInstance(report, extensions.PrintReport)


class TestPrintReport(unittest.TestCase):
    def _setup(self, delete_flush=False):
        self.logreport = MagicMock(spec=extensions.LogReport(
            ['epoch'], trigger=(1, 'iteration'), log_name=None))
        self.stream = MagicMock()
        if delete_flush:
            del self.stream.flush
        self.report = extensions.PrintReport(
            ['epoch'], log_report=self.logreport, out=self.stream)

        self.trainer = testing.get_trainer_with_mock_updater(
            stop_trigger=(1, 'iteration'))
        self.trainer.extend(self.logreport)
        self.trainer.extend(self.report)
        self.logreport.log = [{'epoch': 0}]

    def test_stream_with_flush_is_flushed(self):
        self._setup(delete_flush=False)
        self.assertTrue(hasattr(self.stream, 'flush'))
        self.stream.flush.assert_not_called()
        self.report(self.trainer)
        self.stream.flush.assert_called_once_with()

    def test_stream_without_flush_is_not_flushed(self):
        self._setup(delete_flush=True)
        self.assertFalse(hasattr(self.stream, 'flush'))
        self.stream.flush = MagicMock()
        self.stream.flush.assert_not_called()
        self.report(self.trainer)
        self.stream.flush.assert_not_called()


testing.run_module(__name__, __file__)
