import tempfile
import unittest

import mock

from chainer import testing
from chainer.training import extensions


class TestPrintReport(unittest.TestCase):
    def _setup(self, stream=None, delete_flush=False):
        self.logreport = mock.MagicMock(spec=extensions.LogReport(
            ['epoch'], trigger=(1, 'iteration'), log_name=None))
        if stream is None:
            self.stream = mock.MagicMock()
            if delete_flush:
                del self.stream.flush
        else:
            self.stream = stream
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
        self.stream.flush.assert_called_with()

    def test_stream_without_flush_raises_no_exception(self):
        self._setup(delete_flush=True)
        self.assertFalse(hasattr(self.stream, 'flush'))
        self.report(self.trainer)

    def test_real_stream_raises_no_exception(self):
        with tempfile.TemporaryFile(mode='w') as stream:
            self._setup(stream=stream)
            self.report(self.trainer)


testing.run_module(__name__, __file__)
