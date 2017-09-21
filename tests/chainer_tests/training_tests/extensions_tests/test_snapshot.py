import unittest

import mock

from chainer import testing
from chainer.training import extensions


class TestSnapshotObject(unittest.TestCase):

    def test_trigger(self):
        target = mock.MagicMock()
        snapshot_object = extensions.snapshot_object(target, 'myfile.dat')
        self.assertEqual(snapshot_object.trigger, (1, 'epoch',))


class TestSnapshot(unittest.TestCase):

    def test_trigger(self):
        snapshot = extensions.snapshot()
        self.assertEqual(snapshot.trigger, (1, 'epoch'))


testing.run_module(__name__, __file__)
