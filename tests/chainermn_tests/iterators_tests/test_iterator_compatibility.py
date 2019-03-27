# This test is based on Chainer's iterator compatibility test.
# The major changed point is that we do not test
#   the order SerialIterator -> MultiNodeIterator,
# because slave iterator must synchronize the batch order with master
# thus should not accept overwriting the batch order by serialization.
# See: chainer/tests/chainer_tests/
#      iterators_tests/test_iterator_compatibility.py (7e8f6cc)

import numpy
import platform
import pytest
import unittest

import chainer
import chainer.testing
import chainermn


class DummySerializer(chainer.serializer.Serializer):

    def __init__(self, target):
        super(DummySerializer, self).__init__()
        self.target = target

    def __getitem__(self, key):
        raise NotImplementedError

    def __call__(self, key, value):
        self.target[key] = value
        return self.target[key]


class DummyDeserializer(chainer.serializer.Deserializer):

    def __init__(self, target):
        super(DummyDeserializer, self).__init__()
        self.target = target

    def __getitem__(self, key):
        raise NotImplementedError

    def __call__(self, key, value):
        if value is None:
            value = self.target[key]
        elif isinstance(value, numpy.ndarray):
            numpy.copyto(value, self.target[key])
        else:
            value = type(value)(numpy.asarray(self.target[key]))
        return value


@chainer.testing.parameterize(*chainer.testing.product({
    'iterator_class': [
        chainer.iterators.SerialIterator,
        chainer.iterators.MultiprocessIterator,
    ],
}))
class TestIteratorCompatibility(unittest.TestCase):

    def setUp(self):
        if self.iterator_class == chainer.iterators.MultiprocessIterator and \
                int(platform.python_version_tuple()[0]) < 3:
            pytest.skip('This test requires Python version >= 3')
        self.communicator = chainermn.create_communicator('naive')

        if self.communicator.size < 2:
            pytest.skip('This test is for multinode only')

        self.N = 6
        self.dataset = numpy.arange(self.N).astype(numpy.float32)
        self.bs = 2

    def test_multi_node_iterator_compatibility(self):
        iters = (
            lambda: chainermn.iterators.create_multi_node_iterator(
                self.iterator_class(
                    self.dataset, batch_size=self.bs),
                self.communicator),
            lambda: self.iterator_class(
                self.dataset, batch_size=self.bs),
        )

        bs_n_ratio = 1. * self.bs / self.N

        it_before, it_after = iters

        it = it_before()

        self.assertEqual(it.epoch, 0)
        self.assertAlmostEqual(it.epoch_detail, 0 * bs_n_ratio)
        batch1 = it.next()
        self.assertEqual(len(batch1), self.bs)
        self.assertIsInstance(batch1, list)
        self.assertFalse(it.is_new_epoch)
        self.assertAlmostEqual(it.epoch_detail, 1 * bs_n_ratio)
        batch2 = it.next()
        self.assertEqual(len(batch2), self.bs)
        self.assertIsInstance(batch2, list)
        self.assertFalse(it.is_new_epoch)
        self.assertAlmostEqual(it.epoch_detail, 2 * bs_n_ratio)

        target = dict()
        it.serialize(DummySerializer(target))

        it = it_after()
        it.serialize(DummyDeserializer(target))
        self.assertFalse(it.is_new_epoch)
        self.assertAlmostEqual(it.epoch_detail, 2 * bs_n_ratio)

        batch3 = it.next()
        self.assertEqual(len(batch3), self.bs)
        self.assertIsInstance(batch3, list)
        self.assertTrue(it.is_new_epoch)
        self.assertEqual(
            sorted(batch1 + batch2 + batch3),
            self.dataset.tolist())
        self.assertAlmostEqual(it.epoch_detail, 3 * bs_n_ratio)

    def test_synchronized_iterator_compatibility(self):
        """
        Do not use `chainer.testing.parameterize` to share the code with
        `test_multi_node_iterator_compatibility` because pytest cannot
        guarantee the execution order of tests produced by `parameterize`,
        which causes unexpected behaviors with MPI programs.
        """
        iters = (
            lambda: chainermn.iterators.create_synchronized_iterator(
                self.iterator_class(
                    self.dataset, batch_size=self.bs),
                self.communicator),
            lambda: self.iterator_class(
                self.dataset, batch_size=self.bs),
        )

        bs_n_ratio = 1. * self.bs / self.N

        it_before, it_after = iters

        it = it_before()

        self.assertEqual(it.epoch, 0)
        self.assertAlmostEqual(it.epoch_detail, 0 * bs_n_ratio)
        batch1 = it.next()
        self.assertEqual(len(batch1), self.bs)
        self.assertIsInstance(batch1, list)
        self.assertFalse(it.is_new_epoch)
        self.assertAlmostEqual(it.epoch_detail, 1 * bs_n_ratio)
        batch2 = it.next()
        self.assertEqual(len(batch2), self.bs)
        self.assertIsInstance(batch2, list)
        self.assertFalse(it.is_new_epoch)
        self.assertAlmostEqual(it.epoch_detail, 2 * bs_n_ratio)

        target = dict()
        it.serialize(DummySerializer(target))

        it = it_after()
        it.serialize(DummyDeserializer(target))
        self.assertFalse(it.is_new_epoch)
        self.assertAlmostEqual(it.epoch_detail, 2 * bs_n_ratio)

        batch3 = it.next()
        self.assertEqual(len(batch3), self.bs)
        self.assertIsInstance(batch3, list)
        self.assertTrue(it.is_new_epoch)
        self.assertEqual(
            sorted(batch1 + batch2 + batch3),
            self.dataset.tolist())
        self.assertAlmostEqual(it.epoch_detail, 3 * bs_n_ratio)
