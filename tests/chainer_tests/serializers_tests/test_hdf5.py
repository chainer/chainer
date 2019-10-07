import os
import sys
import tempfile
import unittest

import mock
import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import link
from chainer import links
from chainer import optimizers
from chainer.serializers import hdf5
from chainer import testing
from chainer.testing import attr
import chainerx


if hdf5._available:
    import h5py


# The tests call `fd, path = tempfile.mkstemp(); os.close(fd)` rather than
# `with tempfile.TemporaryFile() as f:` because the file-like objects support
# for `h5py.File` is from h5py>=2.9 (h5py/h5py#1061). h5py>=2.5 is supported.
#
# `os.remove(path)` is necessary. The tests could utilize
# `tempfile.NamedTemporaryFile` but cannot utilize its with-blocks because it
# is platform-dependent behavior to open `f.name` before `f.file` is closed.


@unittest.skipUnless(hdf5._available, 'h5py is not available')
class TestHDF5Serializer(unittest.TestCase):

    def setUp(self):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        self.temp_file_path = path
        self.hdf5file = h5py.File(path, 'w')
        self.serializer = hdf5.HDF5Serializer(self.hdf5file, compression=3)

        self.data = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)

    def tearDown(self):
        if hasattr(self, 'hdf5file'):
            self.hdf5file.close()
        if hasattr(self, 'temp_file_path'):
            os.remove(self.temp_file_path)

    def test_get_item(self):
        child = self.serializer['x']
        self.assertIsInstance(child, hdf5.HDF5Serializer)
        self.assertEqual(child.group.name, '/x')
        self.assertEqual(child.compression, 3)

    def check_serialize(self, data):
        ret = self.serializer('w', data)
        dset = self.hdf5file['w']

        self.assertIsInstance(dset, h5py.Dataset)
        self.assertEqual(dset.shape, data.shape)
        self.assertEqual(dset.size, data.size)
        self.assertEqual(dset.dtype, data.dtype)
        read = numpy.empty((2, 3), dtype=numpy.float32)
        dset.read_direct(read)
        numpy.testing.assert_array_equal(read, backend.CpuDevice().send(data))

        self.assertEqual(dset.compression_opts, 3)

        self.assertIs(ret, data)

    @attr.chainerx
    def test_serialize_chainerx(self):
        self.check_serialize(chainerx.asarray(self.data))

    def test_serialize_cpu(self):
        self.check_serialize(self.data)

    @attr.gpu
    def test_serialize_gpu(self):
        self.check_serialize(cuda.to_gpu(self.data))

    def test_serialize_scalar(self):
        ret = self.serializer('x', 10)
        dset = self.hdf5file['x']

        self.assertIsInstance(dset, h5py.Dataset)
        self.assertEqual(dset.shape, ())
        self.assertEqual(dset.size, 1)
        self.assertEqual(dset.dtype, int)
        read = numpy.empty((), dtype=numpy.int32)
        dset.read_direct(read)
        self.assertEqual(read, 10)

        self.assertEqual(dset.compression_opts, None)

        self.assertIs(ret, 10)

    @unittest.skipUnless(hdf5._available and
                         h5py.version.version_tuple >= (2, 7, 0),
                         'h5py>=2.7.0 is not available')
    def test_serialize_none(self):
        ret = self.serializer('x', None)
        self.assertIs(ret, None)

        dset = self.hdf5file['x']
        self.assertIsInstance(dset, h5py.Dataset)
        self.assertIs(dset.shape, None)


@unittest.skipUnless(hdf5._available, 'h5py is not available')
class TestHDF5Deserializer(unittest.TestCase):

    def setUp(self):
        self.data = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)

        fd, path = tempfile.mkstemp()
        os.close(fd)
        self.temp_file_path = path
        with h5py.File(path, 'w') as f:
            f.require_group('x')
            f.create_dataset('y', data=self.data)
            f.create_dataset('z', data=numpy.asarray(10))
            # h5py.Empty is introduced from 2.7.0
            if h5py.version.version_tuple >= (2, 7, 0):
                f.create_dataset('w', data=h5py.Empty('f'))

        self.hdf5file = h5py.File(path, 'r')
        self.deserializer = hdf5.HDF5Deserializer(self.hdf5file)

    def tearDown(self):
        if hasattr(self, 'hdf5file'):
            self.hdf5file.close()
        if hasattr(self, 'temp_file_path'):
            os.remove(self.temp_file_path)

    def test_get_item(self):
        child = self.deserializer['x']
        self.assertIsInstance(child, hdf5.HDF5Deserializer)
        self.assertEqual(child.group.name, '/x')

    def check_deserialize(self, y):
        ret = self.deserializer('y', y)
        numpy.testing.assert_array_equal(backend.CpuDevice().send(y),
                                         self.data)
        self.assertIs(ret, y)

    def check_deserialize_none_value(self, y):
        ret = self.deserializer('y', None)
        numpy.testing.assert_array_equal(backend.CpuDevice().send(ret),
                                         self.data)

    @attr.chainerx
    def test_deserialize_chainerx(self):
        y = numpy.empty((2, 3), dtype=numpy.float32)
        self.check_deserialize(chainerx.asarray(y))

    @attr.chainerx
    @attr.gpu
    def test_deserialize_chainerx_non_native(self):
        y = numpy.empty((2, 3), dtype=numpy.float32)
        self.check_deserialize(chainerx.asarray(y, device='cuda:0'))

    def test_deserialize_cpu(self):
        y = numpy.empty((2, 3), dtype=numpy.float32)
        self.check_deserialize(y)

    def test_deserialize_none_value_cpu(self):
        y = numpy.empty((2, 3), dtype=numpy.float32)
        self.check_deserialize_none_value(y)

    @attr.gpu
    def test_deserialize_gpu(self):
        y = numpy.empty((2, 3), dtype=numpy.float32)
        self.check_deserialize(cuda.to_gpu(y))

    @attr.ideep
    def test_deserialize_ideep(self):
        y = numpy.empty((2, 3), dtype=numpy.float32)
        self.check_deserialize(intel64.mdarray(y))

    @attr.gpu
    def test_deserialize_none_value_gpu(self):
        y = numpy.empty((2, 3), dtype=numpy.float32)
        self.check_deserialize_none_value(cuda.to_gpu(y))

    def test_deserialize_different_dtype_cpu(self):
        y = numpy.empty((2, 3), dtype=numpy.float16)
        ret = self.deserializer('y', y)

        self.assertIs(ret, y)

        # Compare the value with the original array. Note that it's not always
        # bit-identical to the result of numpy.ndarray.astype.
        numpy.testing.assert_allclose(
            y, self.data.astype(numpy.float16),
            rtol=1e-3, atol=1e-3)

        # It should be bit-identical to the result directly retrieved from
        # h5py.
        arr_hdf5 = numpy.empty((2, 3), dtype=numpy.float16)
        fd, path = tempfile.mkstemp()
        os.close(fd)
        try:
            with h5py.File(path, 'w') as f:
                f.create_dataset('a', data=self.data)
                f['a'].read_direct(arr_hdf5)
        finally:
            os.remove(path)
        numpy.testing.assert_array_equal(y, arr_hdf5)

    @attr.gpu
    def test_deserialize_different_dtype_gpu(self):
        y = cuda.cupy.empty((2, 3), dtype=numpy.float16)
        ret = self.deserializer('y', y)
        numpy.testing.assert_array_equal(
            y.get(), self.data.astype(numpy.float16))
        self.assertIs(ret, y)

    def test_deserialize_scalar(self):
        z = 5
        ret = self.deserializer('z', z)
        self.assertEqual(ret, 10)

    def test_string(self):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        try:
            data = 'abc'
            with h5py.File(path, 'w') as f:
                f.create_dataset('str', data=data)
            with h5py.File(path, 'r') as f:
                deserializer = hdf5.HDF5Deserializer(f)
                ret = deserializer('str', '')
                self.assertEqual(ret, data)
        finally:
            os.remove(path)

    @unittest.skipUnless(hdf5._available and
                         h5py.version.version_tuple >= (2, 7, 0),
                         'h5py>=2.7.0 is not available')
    def test_deserialize_none(self):
        ret = self.deserializer('w', None)
        self.assertIs(ret, None)

    @unittest.skipUnless(hdf5._available and
                         h5py.version.version_tuple >= (2, 7, 0),
                         'h5py>=2.7.0 is not available')
    def test_deserialize_none_by_passing_array(self):
        y = numpy.empty((1,))
        ret = self.deserializer('w', y)
        self.assertIs(ret, None)


@unittest.skipUnless(hdf5._available, 'h5py is not available')
class TestHDF5DeserializerNonStrict(unittest.TestCase):

    def setUp(self):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        self.temp_file_path = path
        with h5py.File(path, 'w') as f:
            f.require_group('x')

        self.hdf5file = h5py.File(path, 'r')
        self.deserializer = hdf5.HDF5Deserializer(self.hdf5file, strict=False)

    def tearDown(self):
        if hasattr(self, 'hdf5file'):
            self.hdf5file.close()
        if hasattr(self, 'temp_file_path'):
            os.remove(self.temp_file_path)

    def test_deserialize_partial(self):
        y = numpy.empty((2, 3), dtype=numpy.float32)
        ret = self.deserializer('y', y)
        self.assertIs(ret, y)


@unittest.skipUnless(hdf5._available, 'h5py is not available')
class TestHDF5DeserializerNonStrictGroupHierachy(unittest.TestCase):

    def setUp(self):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        self.temp_file_path = path

        child = link.Chain()
        with child.init_scope():
            child.linear = links.Linear(2, 3)
        parent = link.Chain()
        with parent.init_scope():
            parent.linear = links.Linear(3, 2)
            parent.child = child
        hdf5.save_hdf5(self.temp_file_path, parent)
        self.source = parent

        self.hdf5file = h5py.File(path, 'r')
        self.deserializer = hdf5.HDF5Deserializer(self.hdf5file, strict=False)

    def tearDown(self):
        if hasattr(self, 'hdf5file'):
            self.hdf5file.close()
        if hasattr(self, 'temp_file_path'):
            os.remove(self.temp_file_path)

    def test_deserialize_hierarchy(self):
        child = link.Chain()
        with child.init_scope():
            child.linear2 = links.Linear(2, 3)
        target = link.Chain()
        with target.init_scope():
            target.linear = links.Linear(3, 2)
            target.child = child
        target_child_W = numpy.copy(child.linear2.W.data)
        target_child_b = numpy.copy(child.linear2.b.data)
        self.deserializer.load(target)

        numpy.testing.assert_array_equal(
            self.source.linear.W.data, target.linear.W.data)
        numpy.testing.assert_array_equal(
            self.source.linear.b.data, target.linear.b.data)
        numpy.testing.assert_array_equal(
            target.child.linear2.W.data, target_child_W)
        numpy.testing.assert_array_equal(
            target.child.linear2.b.data, target_child_b)


@unittest.skipUnless(hdf5._available, 'h5py is not available')
class TestSaveHDF5(unittest.TestCase):

    def setUp(self):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        self.temp_file_path = path

    def tearDown(self):
        if hasattr(self, 'temp_file_path'):
            os.remove(self.temp_file_path)

    def test_save(self):
        obj = mock.MagicMock()
        hdf5.save_hdf5(self.temp_file_path, obj, compression=3)

        self.assertEqual(obj.serialize.call_count, 1)
        (serializer,), _ = obj.serialize.call_args
        self.assertIsInstance(serializer, hdf5.HDF5Serializer)
        self.assertEqual(serializer.compression, 3)


@unittest.skipUnless(hdf5._available, 'h5py is not available')
class TestLoadHDF5(unittest.TestCase):

    def setUp(self):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        self.temp_file_path = path
        # Make a hdf5 file with empty data
        h5py.File(path, 'w')

    def tearDown(self):
        if hasattr(self, 'temp_file_path'):
            os.remove(self.temp_file_path)

    def test_load(self):
        obj = mock.MagicMock()
        hdf5.load_hdf5(self.temp_file_path, obj)

        self.assertEqual(obj.serialize.call_count, 1)
        (serializer,), _ = obj.serialize.call_args
        self.assertIsInstance(serializer, hdf5.HDF5Deserializer)


@unittest.skipUnless(hdf5._available, 'h5py is not available')
class TestGroupHierachy(unittest.TestCase):

    def setUp(self):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        self.temp_file_path = path

        child = link.Chain()
        with child.init_scope():
            child.linear = links.Linear(2, 3)
            child.Wc = chainer.Parameter(shape=(2, 3))

        self.parent = link.Chain()
        with self.parent.init_scope():
            self.parent.child = child
            self.parent.Wp = chainer.Parameter(shape=(2, 3))

        self.optimizer = optimizers.AdaDelta()
        self.optimizer.setup(self.parent)

        self.parent.cleargrads()
        self.optimizer.update()  # init states

    def _save(self, h5, obj, name):
        group = h5.create_group(name)
        serializer = hdf5.HDF5Serializer(group)
        serializer.save(obj)

    def _load(self, h5, obj, name):
        group = h5[name]
        serializer = hdf5.HDF5Deserializer(group)
        serializer.load(obj)

    def tearDown(self):
        if hasattr(self, 'temp_file_path'):
            os.remove(self.temp_file_path)

    def _check_group(self, h5, state):
        self.assertSetEqual(set(h5.keys()),
                            set(('child',) + state))
        self.assertSetEqual(set(h5['child'].keys()),
                            {'linear', 'Wc'})
        self.assertSetEqual(set(h5['child']['linear'].keys()),
                            {'W', 'b'})

    def test_save_chain(self):
        with h5py.File(self.temp_file_path, 'w') as h5:
            self._save(h5, self.parent, 'test')
            self.assertSetEqual(set(h5.keys()), {'test'})
            self._check_group(h5['test'], ('Wp',))

    def test_save_optimizer(self):
        with h5py.File(self.temp_file_path, 'w') as h5:
            self._save(h5, self.optimizer, 'test')
            self.assertSetEqual(set(h5.keys()), {'test'})
            self._check_group(h5['test'], ('Wp', 'epoch', 't'))

    def test_save_chain2(self):
        hdf5.save_hdf5(self.temp_file_path, self.parent)
        with h5py.File(self.temp_file_path, 'r') as h5:
            self._check_group(h5, ('Wp',))

    def test_save_optimizer2(self):
        hdf5.save_hdf5(self.temp_file_path, self.optimizer)
        with h5py.File(self.temp_file_path, 'r') as h5:
            self._check_group(h5, ('Wp', 'epoch', 't'))

    def test_load_chain(self):
        with h5py.File(self.temp_file_path, 'w') as h5:
            self._save(h5, self.parent, 'test')

        with h5py.File(self.temp_file_path, 'r') as h5:
            self._load(h5, self.parent, 'test')

    def test_load_optimizer(self):
        with h5py.File(self.temp_file_path, 'w') as h5:
            self._save(h5, self.optimizer, 'test')

        with h5py.File(self.temp_file_path, 'r') as h5:
            self._load(h5, self.optimizer, 'test')


@unittest.skipUnless(hdf5._available, 'h5py is not available')
class TestNoH5py(unittest.TestCase):

    def setUp(self):
        # Remove h5py from sys.modules to emulate situation that h5py is not
        # installed.
        sys.modules['h5py'] = None

    def tearDown(self):
        sys.modules['h5py'] = h5py

    def test_raise(self):
        del sys.modules['chainer.serializers.hdf5']
        del sys.modules['chainer.serializers.npz']
        del sys.modules['chainer.serializers']

        import chainer.serializers
        self.assertFalse(chainer.serializers.hdf5._available)
        with self.assertRaises(RuntimeError):
            chainer.serializers.save_hdf5(None, None, None)
        with self.assertRaises(RuntimeError):
            chainer.serializers.load_hdf5(None, None)
        with self.assertRaises(RuntimeError):
            chainer.serializers.HDF5Serializer(None)
        with self.assertRaises(RuntimeError):
            chainer.serializers.HDF5Deserializer(None)


@unittest.skipUnless(hdf5._available, 'h5py is not available')
class Test5pyEmptyNotAvailable(unittest.TestCase):

    def setUp(self):
        # Set h5py.version.version_tuple to emulate situation that h5py is
        # so old that it doesn't have h5py.Empty.
        self.original_version_tuple = h5py.version.version_tuple
        h5py.version.version_tuple = (2, 6, 0)

        # Prepare serializer
        fd, path = tempfile.mkstemp()
        os.close(fd)
        self.temp_file_path = path
        self.hdf5file = h5py.File(path, 'w')
        self.serializer = hdf5.HDF5Serializer(self.hdf5file, compression=3)

    def tearDown(self):
        h5py.version.version_tuple = self.original_version_tuple

    def test_raise1(self):
        with self.assertRaises(RuntimeError):
            self.serializer('x', None)


testing.run_module(__name__, __file__)
