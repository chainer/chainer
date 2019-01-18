import os
import tempfile
import unittest

import mock
import numpy
import six

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import link
from chainer import links
from chainer import optimizers
from chainer.serializers import npz
from chainer import testing
from chainer.testing import attr
import chainerx


class TestDictionarySerializer(unittest.TestCase):

    def setUp(self):
        self.serializer = npz.DictionarySerializer({})

        self.data = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)

    def test_get_item(self):
        child = self.serializer['x']
        self.assertIsInstance(child, npz.DictionarySerializer)
        self.assertEqual(child.path, 'x/')

    def test_get_item_strip_slashes(self):
        child = self.serializer['/x/']
        self.assertEqual(child.path, 'x/')

    def check_serialize(self, data, query):
        ret = self.serializer(query, data)
        dset = self.serializer.target['w']

        self.assertIsInstance(dset, numpy.ndarray)
        self.assertEqual(dset.shape, data.shape)
        self.assertEqual(dset.size, data.size)
        self.assertEqual(dset.dtype, data.dtype)
        numpy.testing.assert_array_equal(dset, backend.CpuDevice().send(data))

        self.assertIs(ret, data)

    @attr.chainerx
    def test_serialize_chainerx(self):
        self.check_serialize(chainerx.asarray(self.data), 'w')

    def test_serialize_cpu(self):
        self.check_serialize(self.data, 'w')

    @attr.gpu
    def test_serialize_gpu(self):
        self.check_serialize(cuda.to_gpu(self.data), 'w')

    def test_serialize_cpu_strip_slashes(self):
        self.check_serialize(self.data, '/w')

    @attr.gpu
    def test_serialize_gpu_strip_slashes(self):
        self.check_serialize(cuda.to_gpu(self.data), '/w')

    def test_serialize_scalar(self):
        ret = self.serializer('x', 10)
        dset = self.serializer.target['x']

        self.assertIsInstance(dset, numpy.ndarray)
        self.assertEqual(dset.shape, ())
        self.assertEqual(dset.size, 1)
        self.assertEqual(dset.dtype, int)
        self.assertEqual(dset[()], 10)

        self.assertIs(ret, 10)

    def test_serialize_none(self):
        ret = self.serializer('x', None)
        dset = self.serializer.target['x']

        self.assertIsInstance(dset, numpy.ndarray)
        self.assertEqual(dset.shape, ())
        self.assertEqual(dset.dtype, numpy.object)
        self.assertIs(dset[()], None)

        self.assertIs(ret, None)


@testing.parameterize(*testing.product({'compress': [False, True]}))
class TestNpzDeserializer(unittest.TestCase):

    def setUp(self):
        self.data = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)

        fd, path = tempfile.mkstemp()
        os.close(fd)
        self.temp_file_path = path
        with open(path, 'wb') as f:
            savez = numpy.savez_compressed if self.compress else numpy.savez
            savez(
                f, **{'x/': None, 'y': self.data, 'z': numpy.asarray(10),
                      'w': None})

        self.npzfile = numpy.load(path)
        self.deserializer = npz.NpzDeserializer(self.npzfile)

    def tearDown(self):
        if hasattr(self, 'npzfile'):
            self.npzfile.close()
        if hasattr(self, 'temp_file_path'):
            os.remove(self.temp_file_path)

    def test_get_item(self):
        child = self.deserializer['x']
        self.assertIsInstance(child, npz.NpzDeserializer)
        self.assertEqual(child.path[-2:], 'x/')

    def test_get_item_strip_slashes(self):
        child = self.deserializer['/x/']
        self.assertEqual(child.path, 'x/')

    def check_deserialize(self, y, query):
        ret = self.deserializer(query, y)
        numpy.testing.assert_array_equal(
            backend.CpuDevice().send(y), self.data)
        self.assertIs(ret, y)

    def check_deserialize_by_passing_none(self, y, query):
        ret = self.deserializer(query, None)
        numpy.testing.assert_array_equal(
            backend.CpuDevice().send(ret), self.data)

    @attr.chainerx
    def test_deserialize_chainerx(self):
        y = numpy.empty((2, 3), dtype=numpy.float32)
        self.check_deserialize(chainerx.asarray(y), 'y')

    def test_deserialize_cpu(self):
        y = numpy.empty((2, 3), dtype=numpy.float32)
        self.check_deserialize(y, 'y')

    def test_deserialize_by_passing_none_cpu(self):
        y = numpy.empty((2, 3), dtype=numpy.float32)
        self.check_deserialize_by_passing_none(y, 'y')

    @attr.gpu
    def test_deserialize_gpu(self):
        y = numpy.empty((2, 3), dtype=numpy.float32)
        self.check_deserialize(cuda.to_gpu(y), 'y')

    @attr.ideep
    def test_deserialize_ideep(self):
        y = numpy.empty((2, 3), dtype=numpy.float32)
        self.check_deserialize(intel64.mdarray(y), 'y')

    @attr.gpu
    def test_deserialize_by_passing_none_gpu(self):
        y = numpy.empty((2, 3), dtype=numpy.float32)
        self.check_deserialize_by_passing_none(cuda.to_gpu(y), 'y')

    def test_deserialize_cpu_strip_slashes(self):
        y = numpy.empty((2, 3), dtype=numpy.float32)
        self.check_deserialize(y, '/y')

    @attr.gpu
    def test_deserialize_gpu_strip_slashes(self):
        y = numpy.empty((2, 3), dtype=numpy.float32)
        self.check_deserialize(cuda.to_gpu(y), '/y')

    def test_deserialize_different_dtype_cpu(self):
        y = numpy.empty((2, 3), dtype=numpy.float16)
        ret = self.deserializer('y', y)
        numpy.testing.assert_array_equal(y, self.data.astype(numpy.float16))
        self.assertIs(ret, y)

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

    def test_deserialize_none(self):
        ret = self.deserializer('w', None)
        self.assertIs(ret, None)

    def test_deserialize_by_passing_array(self):
        y = numpy.empty((1,), dtype=numpy.float32)
        ret = self.deserializer('w', y)
        self.assertIs(ret, None)


class TestNpzDeserializerNonStrict(unittest.TestCase):

    def setUp(self):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        self.temp_file_path = path
        with open(path, 'wb') as f:
            numpy.savez(
                f, **{'x': numpy.asarray(10)})

        self.npzfile = numpy.load(path)
        self.deserializer = npz.NpzDeserializer(self.npzfile, strict=False)

    def tearDown(self):
        if hasattr(self, 'npzfile'):
            self.npzfile.close()
        if hasattr(self, 'temp_file_path'):
            os.remove(self.temp_file_path)

    def test_deserialize_partial(self):
        y = numpy.empty((2, 3), dtype=numpy.float32)
        ret = self.deserializer('y', y)
        self.assertIs(ret, y)


@testing.parameterize(
    {'ignore_names': 'yy'},
    {'ignore_names': ['yy']},
    {'ignore_names': lambda key: key == 'yy'},
    {'ignore_names': [lambda key: key == 'yy']},
)
class TestNpzDeserializerIgnoreNames(unittest.TestCase):

    def setUp(self):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        self.temp_file_path = path
        with open(path, 'wb') as f:
            numpy.savez(
                f, **{'x': numpy.asarray(10), 'yy': numpy.empty((2, 3))})

        self.npzfile = numpy.load(path)
        self.deserializer = npz.NpzDeserializer(
            self.npzfile, ignore_names=self.ignore_names)

    def tearDown(self):
        if hasattr(self, 'npzfile'):
            self.npzfile.close()
        if hasattr(self, 'temp_file_path'):
            os.remove(self.temp_file_path)

    def test_deserialize_ignore_names(self):
        yy = numpy.ones((2, 1), dtype=numpy.float32)
        ret = self.deserializer('yy', yy)
        self.assertIs(ret, yy)


@testing.parameterize(
    {'ignore_names': 'yy'},
    {'ignore_names': ['yy']},
    {'ignore_names': lambda key: key == 'yy'},
    {'ignore_names': [lambda key: key == 'yy']},
)
class TestLoadNpzIgnoreNames(unittest.TestCase):

    def setUp(self):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        self.temp_file_path = path
        self.x = numpy.asarray(10, dtype=numpy.float32)
        self.yy = numpy.ones((2, 3), dtype=numpy.float32)
        with open(path, 'wb') as f:
            numpy.savez(
                f, **{'x': self.x, 'yy': self.yy})

    def tearDown(self):
        if hasattr(self, 'temp_file_path'):
            os.remove(self.temp_file_path)

    def test_load_npz_ignore_names(self):
        chain = link.Chain()
        with chain.init_scope():
            chain.x = chainer.variable.Parameter(shape=())
            chain.yy = chainer.variable.Parameter(shape=(2, 3))
        npz.load_npz(
            self.temp_file_path, chain, ignore_names=self.ignore_names)
        self.assertEqual(chain.x.data, self.x)
        self.assertFalse(numpy.all(chain.yy.data == self.yy))


@testing.parameterize(*testing.product({'file_type': ['filename', 'bytesio']}))
class TestNpzDeserializerNonStrictGroupHierachy(unittest.TestCase):

    def setUp(self):
        if self.file_type == 'filename':
            fd, path = tempfile.mkstemp()
            os.close(fd)
            self.file = path
        elif self.file_type == 'bytesio':
            self.file = six.BytesIO()
        else:
            assert False

        # Create and save a link
        child = link.Chain()
        with child.init_scope():
            child.linear = links.Linear(2, 3)
        parent = link.Chain()
        with parent.init_scope():
            parent.linear = links.Linear(3, 2)
            parent.child = child
        npz.save_npz(self.file, parent)
        self.source = parent

        if self.file_type == 'bytesio':
            self.file.seek(0)

        self.npzfile = numpy.load(self.file)
        self.deserializer = npz.NpzDeserializer(self.npzfile, strict=False)

    def tearDown(self):
        if hasattr(self, 'npzfile'):
            self.npzfile.close()
        if self.file_type == 'filename':
            os.remove(self.file)

    def test_deserialize_hierarchy(self):
        # Load a link
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

        # Check
        numpy.testing.assert_array_equal(
            self.source.linear.W.data, target.linear.W.data)
        numpy.testing.assert_array_equal(
            self.source.linear.W.data, target.linear.W.data)
        numpy.testing.assert_array_equal(
            self.source.linear.b.data, target.linear.b.data)
        numpy.testing.assert_array_equal(
            target.child.linear2.W.data, target_child_W)
        numpy.testing.assert_array_equal(
            target.child.linear2.b.data, target_child_b)


class TestSerialize(unittest.TestCase):

    def test_serialize(self):
        obj = mock.MagicMock()
        target = npz.serialize(obj)

        assert obj.serialize.call_count == 1
        (serializer,), _ = obj.serialize.call_args
        assert isinstance(serializer, npz.DictionarySerializer)
        assert isinstance(target, dict)


@testing.parameterize(
    {'ignore_names': ['linear/W', 'child/linear2/b']},
    {'ignore_names': lambda key: key in ['linear/W', 'child/linear2/b']},
    {'ignore_names': [
        lambda key: key in ['linear/W'],
        lambda key: key in ['child/linear2/b']]},
    {'ignore_names': [
        lambda key: key in ['linear/W'],
        'child/linear2/b']},
)
class TestNpzDeserializerIgnoreNamesGroupHierachy(unittest.TestCase):

    def setUp(self):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        self.temp_file_path = path

        child = link.Chain()
        with child.init_scope():
            child.linear2 = links.Linear(2, 3)
        parent = link.Chain()
        with parent.init_scope():
            parent.linear = links.Linear(3, 2)
            parent.child = child
        npz.save_npz(self.temp_file_path, parent)
        self.source = parent

        self.npzfile = numpy.load(path)
        self.deserializer = npz.NpzDeserializer(
            self.npzfile, ignore_names=self.ignore_names)

    def tearDown(self):
        if hasattr(self, 'npzfile'):
            self.npzfile.close()
        if hasattr(self, 'temp_file_path'):
            os.remove(self.temp_file_path)

    def test_deserialize_ignore_names(self):
        child = link.Chain()
        with child.init_scope():
            child.linear2 = links.Linear(2, 3)
        target = link.Chain()
        with target.init_scope():
            target.linear = links.Linear(3, 2)
            target.child = child
        target_W = numpy.copy(target.linear.W.data)
        target_child_b = numpy.copy(child.linear2.b.data)
        self.deserializer.load(target)

        numpy.testing.assert_array_equal(
            self.source.linear.b.data, target.linear.b.data)
        numpy.testing.assert_array_equal(
            self.source.child.linear2.W.data, target.child.linear2.W.data)
        numpy.testing.assert_array_equal(
            target.linear.W.data, target_W)
        numpy.testing.assert_array_equal(
            target.child.linear2.b.data, target_child_b)


@testing.parameterize(*testing.product({
    'compress': [False, True],
    'file_type': ['filename', 'bytesio'],
}))
class TestSaveNpz(unittest.TestCase):

    def setUp(self):
        if self.file_type == 'filename':
            fd, path = tempfile.mkstemp()
            os.close(fd)
            self.file = path
        elif self.file_type == 'bytesio':
            self.file = six.BytesIO()
        else:
            assert False

    def tearDown(self):
        if self.file_type == 'filename':
            os.remove(self.file)

    def test_save(self):
        obj = mock.MagicMock()
        npz.save_npz(self.file, obj, self.compress)

        self.assertEqual(obj.serialize.call_count, 1)
        (serializer,), _ = obj.serialize.call_args
        self.assertIsInstance(serializer, npz.DictionarySerializer)


@testing.parameterize(*testing.product({
    'compress': [False, True],
    'file_type': ['filename', 'bytesio'],
}))
class TestLoadNpz(unittest.TestCase):

    def setUp(self):
        if self.file_type == 'filename':
            fd, path = tempfile.mkstemp()
            os.close(fd)
            self.file = path
        elif self.file_type == 'bytesio':
            self.file = six.BytesIO()
        else:
            assert False

        child = link.Chain()
        with child.init_scope():
            child.child_linear = links.Linear(2, 3)
        parent = link.Chain()
        with parent.init_scope():
            parent.parent_linear = links.Linear(3, 2)
            parent.child = child
        npz.save_npz(self.file, parent, self.compress)

        if self.file_type == 'bytesio':
            self.file.seek(0)

        self.source_child = child
        self.source_parent = parent

    def tearDown(self):
        if self.file_type == 'filename':
            os.remove(self.file)

    def test_load_with_strict(self):
        obj = mock.MagicMock()
        npz.load_npz(self.file, obj)

        self.assertEqual(obj.serialize.call_count, 1)
        (serializer,), _ = obj.serialize.call_args
        self.assertIsInstance(serializer, npz.NpzDeserializer)
        self.assertTrue(serializer.strict)

    def test_load_without_strict(self):
        obj = mock.MagicMock()
        npz.load_npz(self.file, obj, strict=False)

        self.assertEqual(obj.serialize.call_count, 1)
        (serializer,), _ = obj.serialize.call_args
        self.assertFalse(serializer.strict)
        self.assertIsInstance(serializer, npz.NpzDeserializer)

    def test_load_with_path(self):
        target = link.Chain()
        with target.init_scope():
            target.child_linear = links.Linear(2, 3)
        npz.load_npz(self.file, target, 'child/')
        numpy.testing.assert_array_equal(
            self.source_child.child_linear.W.data, target.child_linear.W.data)

    def test_load_without_path(self):
        target = link.Chain()
        with target.init_scope():
            target.parent_linear = links.Linear(3, 2)
        npz.load_npz(self.file, target, path='')
        numpy.testing.assert_array_equal(
            self.source_parent.parent_linear.W.data,
            target.parent_linear.W.data)


@testing.parameterize(*testing.product({
    'compress': [False, True],
    'file_type': ['filename', 'bytesio'],
}))
class TestGroupHierachy(unittest.TestCase):

    def setUp(self):
        if self.file_type == 'filename':
            fd, path = tempfile.mkstemp()
            os.close(fd)
            self.file = path
        elif self.file_type == 'bytesio':
            self.file = six.BytesIO()
        else:
            assert False

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
        self.optimizer.update()  # init all states

        self.savez = numpy.savez_compressed if self.compress else numpy.savez

    def tearDown(self):
        if self.file_type == 'filename':
            os.remove(self.file)

    def _save(self, target, obj, name):
        serializer = npz.DictionarySerializer(target, name)
        serializer.save(obj)

    def _savez(self, file, d):
        if self.file_type == 'filename':
            f = open(self.file, 'wb')
        elif self.file_type == 'bytesio':
            f = self.file
        else:
            assert False

        self.savez(f, **d)

        if self.file_type == 'bytesio':
            self.file.seek(0)

    def _save_npz(self, file, obj, compress):
        npz.save_npz(file, obj, compress)
        if self.file_type == 'bytesio':
            self.file.seek(0)

    def _check_chain_group(self, npzfile, state, prefix=''):
        keys = ('child/linear/W',
                'child/linear/b',
                'child/Wc') + state
        self.assertSetEqual(set(npzfile.keys()), {prefix + x for x in keys})

    def _check_optimizer_group(self, npzfile, state, prefix=''):
        keys = ('child/linear/W/t',
                'child/linear/W/msg',
                'child/linear/W/msdx',
                'child/linear/b/t',
                'child/linear/b/msg',
                'child/linear/b/msdx',
                'child/Wc/t',
                'child/Wc/msg',
                'child/Wc/msdx') + state
        self.assertEqual(set(npzfile.keys()),
                         {prefix + x for x in keys})

    def test_save_chain(self):
        d = {}
        self._save(d, self.parent, 'test/')
        self._savez(self.file, d)

        with numpy.load(self.file) as f:
            self._check_chain_group(f, ('Wp',), 'test/')

    def test_save_optimizer(self):
        d = {}
        self._save(d, self.optimizer, 'test/')
        self._savez(self.file, d)

        with numpy.load(self.file) as npzfile:
            self._check_optimizer_group(
                npzfile, ('Wp/t', 'Wp/msg', 'Wp/msdx', 'epoch', 't'), 'test/')

    def test_save_chain2(self):
        self._save_npz(self.file, self.parent, self.compress)
        with numpy.load(self.file) as npzfile:
            self._check_chain_group(npzfile, ('Wp',))

    def test_save_optimizer2(self):
        self._save_npz(self.file, self.optimizer, self.compress)
        with numpy.load(self.file) as npzfile:
            self._check_optimizer_group(
                npzfile, ('Wp/t', 'Wp/msg', 'Wp/msdx', 'epoch', 't'))

    def test_load_optimizer_with_strict(self):
        for param in self.parent.params():
            param.data.fill(1)
        self._save_npz(self.file, self.parent, self.compress)
        for param in self.parent.params():
            param.data.fill(0)
        npz.load_npz(self.file, self.parent)
        for param in self.parent.params():
            self.assertTrue((param.data == 1).all())

    def test_load_optimizer_without_strict(self):
        for param in self.parent.params():
            param.data.fill(1)
        self._save_npz(self.file, self.parent, self.compress)
        # Remove a param
        del self.parent.child.linear.b
        for param in self.parent.params():
            param.data.fill(0)
        npz.load_npz(self.file, self.parent, strict=False)
        for param in self.parent.params():
            self.assertTrue((param.data == 1).all())
        self.assertFalse(hasattr(self.parent.child.linear, 'b'))


testing.run_module(__name__, __file__)
