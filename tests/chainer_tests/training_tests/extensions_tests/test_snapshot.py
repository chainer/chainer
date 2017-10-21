import unittest

import mock

from chainer import testing
from chainer.training import extensions


class TestSnapshot(unittest.TestCase):

    def test_call(self):
        t = mock.MagicMock()
        c = mock.MagicMock(side_effect=[True, False])
        w = mock.MagicMock()
        snapshot = extensions.Snapshot(target=t, condition=c, writer=w)
        trainer = mock.MagicMock()
        snapshot(trainer)
        snapshot(trainer)

        self.assertEqual(c.call_count, 2)
        self.assertEqual(w.call_count, 1)


testing.run_module(__name__, __file__)
