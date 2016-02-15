import contextlib
import os
import tempfile
import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer import dataset
from chainer import serializers
from chainer.testing import attr
from chainer.testing import condition


class IncompleteDataset(chainer.Dataset):
    pass


class SimpleDataset(chainer.Dataset):

    name = 'simple_dataset'

    def __init__(self, values):
        self._values = values

    def __len__(self):
        return len(self._values)

    def __getitem__(self, i):
        return self._values[i]


class TestUndefinedMethods(unittest.TestCase):

    def setUp(self):
        self.ds = IncompleteDataset()

    def test_undefined_len(self):
        with self.assertRaises(NotImplementedError):
            len(self.ds)

    def test_undefined_getitem(self):
        with self.assertRaises(NotImplementedError):
            self.ds[0]


class TestSimpleDataset(unittest.TestCase):

    def setUp(self):
        self.values = [numpy.array([i]) for i in range(5)]
        self.ds = SimpleDataset(self.values)

    def test_iter(self):
        it = self.ds.get_batch_iterator(auto_shuffle=False)
        for i in range(len(self.ds)):
            x = it.next()
            self.assertIsInstance(x, numpy.ndarray)
            self.assertEqual(x.shape, (1, 1))
            self.assertEqual(x[0, 0], self.values[i])

    def test_iter_batch(self):
        it = self.ds.get_batch_iterator(batchsize=2, auto_shuffle=False)
        xs = [it.next() for _ in self.values]
        for i, x in enumerate(xs):
            self.assertIsInstance(x, numpy.ndarray)
            self.assertEqual(x.shape, (2, 1))
            v1 = 2 * i % len(self.ds)
            v2 = (2 * i + 1) % len(self.ds)
            numpy.testing.assert_array_equal(x, [[v1], [v2]])

    @condition.retry(10)
    def test_iter_shuffle(self):
        shuffled_it = self.ds.get_batch_iterator(auto_shuffle=False)
        xs0 = [int(shuffled_it.next()) for _ in self.values]

        it = self.ds.get_batch_iterator()
        xs1 = [int(it.next()) for _ in self.values]
        xs2 = [int(it.next()) for _ in self.values]

        self.assertTrue(any([x0 != x1 for x0, x1 in zip(xs0, xs1)]))
        self.assertTrue(any([x0 != x2 for x0, x2 in zip(xs0, xs2)]))
        self.assertTrue(any([x1 != x2 for x1, x2 in zip(xs1, xs2)]))

        xs1.sort()
        xs2.sort()
        self.assertTrue(all([x0 == x1 for x0, x1 in zip(xs0, xs1)]))
        self.assertTrue(all([x0 == x2 for x0, x2 in zip(xs0, xs2)]))

    def test_iter_not_repeated(self):
        it = self.ds.get_batch_iterator(repeat=False)
        for _ in self.values:
            it.next()
        with self.assertRaises(StopIteration):
            it.next()

    @attr.gpu
    def test_iter_gpu_device(self):
        it = self.ds.get_batch_iterator(auto_shuffle=False, device=0)
        for x in self.values:
            y = it.next()
            self.assertIsInstance(y, cuda.ndarray)
            self.assertEqual(y.device.id, 0)
            y_cpu = cuda.to_cpu(y)
            self.assertEqual(int(y_cpu), int(x))

    @attr.gpu
    def test_iter_cpu_device(self):
        for i in range(len(self.values)):
            self.values[i] = cuda.to_gpu(self.values[i])
        it = self.ds.get_batch_iterator(auto_shuffle=False, device=-1)
        for x in self.values:
            y = it.next()
            self.assertIsInstance(y, numpy.ndarray)
            x_cpu = cuda.to_cpu(x)
            self.assertEqual(int(y), int(x_cpu))

    def test_iter_epoch(self):
        it = self.ds.get_batch_iterator()
        for epoch in range(3):
            for x in self.values:
                self.assertEqual(it.epoch, epoch)
                it.next()

    def test_iter_batch_epoch(self):
        it = self.ds.get_batch_iterator(batchsize=2)
        for i in range(10):
            self.assertEqual(it.epoch, i * 2 // len(self.values))
            it.next()

    def test_iter_serialize(self):
        it = self.ds.get_batch_iterator(batchsize=2, auto_shuffle=False)
        jt = self.ds.get_batch_iterator(batchsize=2, auto_shuffle=False)
        it.next()

        with _serialize(it) as path:
            serializers.load_npz(path, jt)

        for i in range(5):
            x = it.next()
            y = jt.next()
            numpy.testing.assert_array_equal(x, y)

    def test_iter_serialize_shuffle(self):
        it = self.ds.get_batch_iterator()
        jt = self.ds.get_batch_iterator()
        it.next()

        with _serialize(it) as path:
            serializers.load_npz(path, jt)

        for i in range(len(self.values) - 1):
            x = it.next()
            y = jt.next()
            numpy.testing.assert_array_equal(x, y)

    def test_iter_serialize_epoch(self):
        it = self.ds.get_batch_iterator(batchsize=2)
        jt = self.ds.get_batch_iterator(batchsize=2)
        for i in range(9):
            it.next()

        with _serialize(it) as path:
            serializers.load_npz(path, jt)

        self.assertEqual(it.epoch, jt.epoch)


class TestTupleDataset(unittest.TestCase):

    def setUp(self):
        self.values = [(numpy.array([i]), numpy.array([i, i]))
                       for i in range(5)]
        self.ds = SimpleDataset(self.values)

    def test_iter(self):
        it = self.ds.get_batch_iterator(auto_shuffle=False)
        for x, y in self.values:
            t = it.next()
            self.assertIsInstance(t, tuple)
            self.assertEqual(len(t), 2)
            numpy.testing.assert_array_equal(t[0], [x])
            numpy.testing.assert_array_equal(t[1], [y])

    def test_iter_batch(self):
        it = self.ds.get_batch_iterator(batchsize=2, auto_shuffle=False)
        for i in range(5):
            t = it.next()
            self.assertIsInstance(t, tuple)
            self.assertEqual(len(t), 2)
            v1 = i * 2 % len(self.values)
            v2 = (i * 2 + 1) % len(self.values)
            numpy.testing.assert_array_equal(t[0], [[v1], [v2]])
            numpy.testing.assert_array_equal(t[1], [[v1, v1], [v2, v2]])


class TestBuildMinibatch(unittest.TestCase):

    def setUp(self):
        self.examples = [(numpy.array([i]), numpy.array([i, i]))
                         for i in range(5)]
        self.expect = (numpy.array([[i] for i in range(5)]),
                       numpy.array([[i, i] for i in range(5)]))

    def test_build_minibatch(self):
        batch = dataset.build_minibatch(self.examples)
        self.assertIsInstance(batch, tuple)
        self.assertEqual(len(batch), 2)
        self.assertIsInstance(batch[0], numpy.ndarray)
        self.assertIsInstance(batch[1], numpy.ndarray)
        numpy.testing.assert_array_equal(batch[0], self.expect[0])
        numpy.testing.assert_array_equal(batch[1], self.expect[1])

    @attr.gpu
    def test_build_minibatch_gpu_device(self):
        cupy = cuda.cupy
        batch = dataset.build_minibatch(self.examples, 0)
        self.assertIsInstance(batch, tuple)
        self.assertEqual(len(batch), 2)
        self.assertIsInstance(batch[0], cupy.ndarray)
        self.assertIsInstance(batch[1], cupy.ndarray)
        cupy.testing.assert_array_equal(batch[0], self.expect[0])
        cupy.testing.assert_array_equal(batch[1], self.expect[1])

    @attr.gpu
    def test_build_minibatch_cpu_device(self):
        examples = [(cuda.to_gpu(x), cuda.to_gpu(y)) for x, y in self.examples]
        batch = dataset.build_minibatch(examples, -1)
        self.assertIsInstance(batch, tuple)
        self.assertEqual(len(batch), 2)
        self.assertIsInstance(batch[0], numpy.ndarray)
        self.assertIsInstance(batch[1], numpy.ndarray)
        numpy.testing.assert_array_equal(batch[0], self.expect[0])
        numpy.testing.assert_array_equal(batch[1], self.expect[1])

    @attr.gpu
    def test_build_minibatch_gpu_pass_through(self):
        cupy = cuda.cupy
        examples = [(cuda.to_gpu(x), cuda.to_gpu(y)) for x, y in self.examples]
        batch = dataset.build_minibatch(examples)
        self.assertIsInstance(batch, tuple)
        self.assertEqual(len(batch), 2)
        self.assertIsInstance(batch[0], cupy.ndarray)
        self.assertIsInstance(batch[1], cupy.ndarray)
        cupy.testing.assert_array_equal(batch[0], self.expect[0])
        cupy.testing.assert_array_equal(batch[1], self.expect[1])

    @attr.multi_gpu(2)
    def test_build_minibatch_gpu_to_another_gpu(self):
        cupy = cuda.cupy
        with cuda.get_device(0):
            examples = [(cuda.to_gpu(x), cuda.to_gpu(y))
                        for x, y in self.examples]
            batch = dataset.build_minibatch(examples, 1)
            self.assertIsInstance(batch, tuple)
            self.assertEqual(len(batch), 2)
            self.assertIsInstance(batch[0], cupy.ndarray)
            self.assertEqual(batch[0].device.id, 1)
            self.assertIsInstance(batch[1], cupy.ndarray)
            self.assertEqual(batch[1].device.id, 1)
            cupy.testing.assert_array_equal(batch[0], self.expect[0])
            cupy.testing.assert_array_equal(batch[1], self.expect[1])


@contextlib.contextmanager
def _serialize(obj):
    _, path = tempfile.mkstemp()
    serializers.save_npz(path, obj)
    yield path
    os.remove(path)
