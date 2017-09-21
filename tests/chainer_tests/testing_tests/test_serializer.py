import unittest

from chainer.serializers import hdf5
from chainer import testing


class Serializable(object):

    def __init__(self, value):
        self.value = value

    def serialize(self, serializer):
        self.value = serializer('value', self.value)


class TestSaveAndLoad(unittest.TestCase):

    def setUp(self):
        self.src = Serializable(1)
        self.dst = Serializable(2)

    def test_save_and_load_npz(self):
        testing.save_and_load_npz(self.src, self.dst)
        self.assertEqual(self.dst.value, 1)

    @unittest.skipUnless(hdf5._available, 'h5py is not available')
    def test_save_and_load_hdf5(self):
        testing.save_and_load_hdf5(self.src, self.dst)
        self.assertEqual(self.dst.value, 1)


testing.run_module(__name__, __file__)
