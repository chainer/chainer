from __future__ import division
import copy
import time
import unittest

import numpy
import six

from chainer import iterators
from chainer import serializer
from chainer import testing


class DummySerializer(serializer.Serializer):

    def __init__(self, target):
        super(DummySerializer, self).__init__()
        self.target = target

    def __getitem__(self, key):
        raise NotImplementedError

    def __call__(self, key, value):
        self.target[key] = value
        return self.target[key]


class DummyDeserializer(serializer.Deserializer):

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


@testing.parameterize(*testing.product({
    'n_threads': [1, 2],
}))
class TestMultithreadIterator(unittest.TestCase):

    def setUp(self):
        self.options = {'n_threads': self.n_threads}

    def test_iterator_repeat(self):
        dataset = [1, 2, 3, 4, 5, 6]
        it = iterators.MultithreadIterator(dataset, 2, **self.options)
        for i in range(3):
            self.assertEqual(it.epoch, i)
            self.assertAlmostEqual(it.epoch_detail, i + 0 / 6)
            if i == 0:
                self.assertIsNone(it.previous_epoch_detail)
            else:
                self.assertAlmostEqual(it.previous_epoch_detail, i - 2 / 6)
            batch1 = it.next()
            self.assertEqual(len(batch1), 2)
            self.assertIsInstance(batch1, list)
            self.assertFalse(it.is_new_epoch)
            self.assertAlmostEqual(it.epoch_detail, i + 2 / 6)
            self.assertAlmostEqual(it.previous_epoch_detail, i + 0 / 6)
            batch2 = it.next()
            self.assertEqual(len(batch2), 2)
            self.assertIsInstance(batch2, list)
            self.assertFalse(it.is_new_epoch)
            self.assertAlmostEqual(it.epoch_detail, i + 4 / 6)
            self.assertAlmostEqual(it.previous_epoch_detail, i + 2 / 6)
            batch3 = it.next()
            self.assertEqual(len(batch3), 2)
            self.assertIsInstance(batch3, list)
            self.assertTrue(it.is_new_epoch)
            self.assertEqual(sorted(batch1 + batch2 + batch3), dataset)
            self.assertAlmostEqual(it.epoch_detail, i + 6 / 6)
            self.assertAlmostEqual(it.previous_epoch_detail, i + 4 / 6)

    def test_iterator_list_type(self):
        dataset = [[i, numpy.zeros((10,)) + i] for i in range(6)]
        it = iterators.MultithreadIterator(dataset, 2, **self.options)
        for i in range(3):
            self.assertEqual(it.epoch, i)
            self.assertAlmostEqual(it.epoch_detail, i)
            if i == 0:
                self.assertIsNone(it.previous_epoch_detail)
            else:
                self.assertAlmostEqual(it.previous_epoch_detail, i - 2 / 6)
            batches = {}
            for j in range(3):
                batch = it.next()
                self.assertEqual(len(batch), 2)
                if j != 2:
                    self.assertFalse(it.is_new_epoch)
                else:
                    self.assertTrue(it.is_new_epoch)
                self.assertAlmostEqual(
                    it.epoch_detail, (3 * i + j + 1) * 2 / 6)
                self.assertAlmostEqual(
                    it.previous_epoch_detail, (3 * i + j) * 2 / 6)
                for x in batch:
                    self.assertIsInstance(x, list)
                    self.assertIsInstance(x[1], numpy.ndarray)
                    batches[x[0]] = x[1]

            self.assertEqual(len(batches), len(dataset))
            for k, v in six.iteritems(batches):
                numpy.testing.assert_allclose(dataset[k][1], v)

    def test_iterator_tuple_type(self):
        dataset = [(i, numpy.zeros((10,)) + i) for i in range(6)]
        it = iterators.MultithreadIterator(dataset, 2, **self.options)
        for i in range(3):
            self.assertEqual(it.epoch, i)
            self.assertAlmostEqual(it.epoch_detail, i)
            if i == 0:
                self.assertIsNone(it.previous_epoch_detail)
            else:
                self.assertAlmostEqual(it.previous_epoch_detail, i - 2 / 6)
            batches = {}
            for j in range(3):
                batch = it.next()
                self.assertEqual(len(batch), 2)
                if j != 2:
                    self.assertFalse(it.is_new_epoch)
                else:
                    self.assertTrue(it.is_new_epoch)
                self.assertAlmostEqual(
                    it.epoch_detail, (3 * i + j + 1) * 2 / 6)
                self.assertAlmostEqual(
                    it.previous_epoch_detail, (3 * i + j) * 2 / 6)
                for x in batch:
                    self.assertIsInstance(x, tuple)
                    self.assertIsInstance(x[1], numpy.ndarray)
                    batches[x[0]] = x[1]

            self.assertEqual(len(batches), len(dataset))
            for k, v in six.iteritems(batches):
                numpy.testing.assert_allclose(dataset[k][1], v)

    def test_iterator_dict_type(self):
        dataset = [{i: numpy.zeros((10,)) + i} for i in range(6)]
        it = iterators.MultithreadIterator(dataset, 2, **self.options)
        for i in range(3):
            self.assertEqual(it.epoch, i)
            self.assertAlmostEqual(it.epoch_detail, i)
            if i == 0:
                self.assertIsNone(it.previous_epoch_detail)
            else:
                self.assertAlmostEqual(it.previous_epoch_detail, i - 2 / 6)
            batches = {}
            for j in range(3):
                batch = it.next()
                self.assertEqual(len(batch), 2)
                if j != 2:
                    self.assertFalse(it.is_new_epoch)
                else:
                    self.assertTrue(it.is_new_epoch)
                self.assertAlmostEqual(
                    it.epoch_detail, (3 * i + j + 1) * 2 / 6)
                self.assertAlmostEqual(
                    it.previous_epoch_detail, (3 * i + j) * 2 / 6)
                for x in batch:
                    self.assertIsInstance(x, dict)
                    k = tuple(x)[0]
                    v = x[k]
                    self.assertIsInstance(v, numpy.ndarray)
                    batches[k] = v

            self.assertEqual(len(batches), len(dataset))
            for k, v in six.iteritems(batches):
                x = dataset[k][tuple(dataset[k])[0]]
                numpy.testing.assert_allclose(x, v)

    def test_iterator_repeat_not_even(self):
        dataset = [1, 2, 3, 4, 5]
        it = iterators.MultithreadIterator(dataset, 2, **self.options)

        batches = sum([it.next() for _ in range(5)], [])
        self.assertEqual(sorted(batches), sorted(dataset * 2))

    def test_iterator_not_repeat(self):
        dataset = [1, 2, 3, 4, 5]
        it = iterators.MultithreadIterator(
            dataset, 2, repeat=False, **self.options)

        batches = sum([it.next() for _ in range(3)], [])
        self.assertEqual(sorted(batches), dataset)
        for _ in range(2):
            self.assertRaises(StopIteration, it.next)

    def test_iterator_not_repeat_not_even(self):
        dataset = [1, 2, 3, 4, 5]
        it = iterators.MultithreadIterator(
            dataset, 2, repeat=False, **self.options)

        self.assertAlmostEqual(it.epoch_detail, 0 / 5)
        self.assertIsNone(it.previous_epoch_detail)
        batch1 = it.next()
        self.assertAlmostEqual(it.epoch_detail, 2 / 5)
        self.assertAlmostEqual(it.previous_epoch_detail, 0 / 5)
        batch2 = it.next()
        self.assertAlmostEqual(it.epoch_detail, 4 / 5)
        self.assertAlmostEqual(it.previous_epoch_detail, 2 / 5)
        batch3 = it.next()
        self.assertAlmostEqual(it.epoch_detail, 5 / 5)
        self.assertAlmostEqual(it.previous_epoch_detail, 4 / 5)
        self.assertRaises(StopIteration, it.next)

        self.assertEqual(len(batch3), 1)
        self.assertEqual(sorted(batch1 + batch2 + batch3), dataset)

    def test_iterator_shuffle_divisible(self):
        dataset = list(range(10))
        it = iterators.MultithreadIterator(
            dataset, 10, **self.options)
        self.assertNotEqual(it.next(), it.next())

    def test_iterator_shuffle_nondivisible(self):
        dataset = list(range(10))
        it = iterators.MultithreadIterator(
            dataset, 3, **self.options)
        out = sum([it.next() for _ in range(7)], [])
        self.assertNotEqual(out[0:10], out[10:20])

    def test_copy_not_repeat(self):
        dataset = [1, 2, 3, 4, 5]
        it = iterators.MultithreadIterator(
            dataset, 2, repeat=False, **self.options)
        copy_it = copy.copy(it)
        batches = sum([it.next() for _ in range(3)], [])
        self.assertEqual(sorted(batches), dataset)
        for _ in range(2):
            self.assertRaises(StopIteration, it.next)
        it = None

        batches = sum([copy_it.next() for _ in range(3)], [])
        self.assertEqual(sorted(batches), dataset)
        for _ in range(2):
            self.assertRaises(StopIteration, copy_it.next)

    def test_reset(self):
        dataset = [1, 2, 3, 4, 5]
        it = iterators.MultithreadIterator(
            dataset, 2, repeat=False, **self.options)

        for trial in range(4):
            batches = sum([it.next() for _ in range(3)], [])
            self.assertEqual(sorted(batches), dataset)
            for _ in range(2):
                self.assertRaises(StopIteration, it.next)
            it.reset()

    def test_supported_reset_middle(self):
        dataset = [1, 2, 3, 4, 5]
        it = iterators.MultithreadIterator(
            dataset, 2, repeat=False, **self.options)
        it.next()
        it.reset()

    def test_supported_reset_repeat(self):
        dataset = [1, 2, 3, 4]
        it = iterators.MultithreadIterator(
            dataset, 2, repeat=True, **self.options)
        it.next()
        it.next()
        it.reset()

    def test_supported_reset_finalized(self):
        dataset = [1, 2, 3, 4]
        it = iterators.MultithreadIterator(
            dataset, 2, repeat=False, **self.options)
        it.next()
        it.next()
        it.finalize()
        it.reset()


@testing.parameterize(*testing.product({
    'n_threads': [1, 2],
}))
class TestMultithreadIteratorSerialize(unittest.TestCase):

    def setUp(self):
        self.options = {'n_threads': self.n_threads}

    def test_iterator_serialize(self):
        dataset = [1, 2, 3, 4, 5, 6]
        it = iterators.MultithreadIterator(dataset, 2, **self.options)

        self.assertEqual(it.epoch, 0)
        self.assertAlmostEqual(it.epoch_detail, 0 / 6)
        self.assertIsNone(it.previous_epoch_detail)
        batch1 = it.next()
        self.assertEqual(len(batch1), 2)
        self.assertIsInstance(batch1, list)
        self.assertFalse(it.is_new_epoch)
        self.assertAlmostEqual(it.epoch_detail, 2 / 6)
        self.assertAlmostEqual(it.previous_epoch_detail, 0 / 6)
        batch2 = it.next()
        self.assertEqual(len(batch2), 2)
        self.assertIsInstance(batch2, list)
        self.assertFalse(it.is_new_epoch)
        self.assertAlmostEqual(it.epoch_detail, 4 / 6)
        self.assertAlmostEqual(it.previous_epoch_detail, 2 / 6)

        target = dict()
        it.serialize(DummySerializer(target))

        it = iterators.MultithreadIterator(dataset, 2, **self.options)
        it.serialize(DummyDeserializer(target))
        self.assertFalse(it.is_new_epoch)
        self.assertAlmostEqual(it.epoch_detail, 4 / 6)
        self.assertAlmostEqual(it.previous_epoch_detail, 2 / 6)

        batch3 = it.next()
        self.assertEqual(len(batch3), 2)
        self.assertIsInstance(batch3, list)
        self.assertTrue(it.is_new_epoch)
        self.assertEqual(sorted(batch1 + batch2 + batch3), dataset)
        self.assertAlmostEqual(it.epoch_detail, 6 / 6)
        self.assertAlmostEqual(it.previous_epoch_detail, 4 / 6)


@testing.parameterize(*testing.product({
    'n_threads': [1, 2],
}))
class TestMultithreadIteratorRandomState(unittest.TestCase):

    def setUp(self):
        self.options = {'shuffle': False,
                        'n_threads': self.n_threads}
        self._seed = 3141592653
        self._random_bak = numpy.random.get_state()

    def tearDown(self):
        numpy.random.set_state(self._random_bak)

    class RandomDataset(object):
        def __init__(self, interleave=False, twice=False):
            self.interleave = interleave
            self.twice = twice

        def __len__(self):
            return 64

        def __getitem__(self, i):
            if self.interleave and i // 4 % 2 == 1:
                return 0
            else:
                random = iterators.get_random_state()
                if self.twice:
                    return random.uniform(), random.uniform()
                else:
                    return random.uniform()

    def test_random_state_different_numbers(self):
        dataset = self.RandomDataset()
        it = iterators.MultithreadIterator(dataset, 4, **self.options)

        batch1 = it.next()
        batch2 = it.next()
        batch3 = it.next()
        batch4 = it.next()

        data = numpy.concatenate([batch1, batch2, batch3, batch4])
        # To prevent accidental failure, we allow two elements to be same.
        self.assertGreaterEqual(numpy.unique(data).size, 15)

    def test_random_state_different_numbers_interleaved(self):
        dataset = self.RandomDataset(interleave=True)
        it = iterators.MultithreadIterator(dataset, 4, **self.options)

        batch1 = it.next()
        it.next()
        batch2 = it.next()
        it.next()
        batch3 = it.next()
        it.next()
        batch4 = it.next()
        it.next()

        data = numpy.concatenate([batch1, batch2, batch3, batch4])
        # To prevent accidental failure, we allow two elements to be same.
        self.assertGreaterEqual(numpy.unique(data).size, 15)

    def test_random_state_same_seed(self):
        dataset1 = self.RandomDataset()
        numpy.random.seed(self._seed)
        it1 = iterators.MultithreadIterator(dataset1, 4, **self.options)

        dataset2 = self.RandomDataset(interleave=True)
        numpy.random.seed(self._seed)
        it2 = iterators.MultithreadIterator(dataset2, 4, **self.options)

        for _ in range(4):
            batch1 = it1.next()
            batch2 = it2.next()
            it2.next()
            self.assertEqual(batch1, batch2)

    def test_random_state_keep_sequence_for_each_batch_idx(self):
        dataset1 = self.RandomDataset()
        numpy.random.seed(self._seed)
        it1 = iterators.MultithreadIterator(dataset1, 4, **self.options)

        dataset2 = self.RandomDataset(twice=True)
        numpy.random.seed(self._seed)
        it2 = iterators.MultithreadIterator(dataset2, 4, **self.options)

        for _ in range(4):
            batch1 = list(zip(it1.next(), it1.next()))
            batch2 = it2.next()
            self.assertEqual(batch1, batch2)

    def test_random_state_reset(self):
        dataset = self.RandomDataset()

        numpy.random.seed(self._seed)
        it1 = iterators.MultithreadIterator(dataset, 4, **self.options)
        it1.next()
        batch_a = it1.next()
        it1.next()
        it1.next()
        batch_b = it1.next()

        numpy.random.seed(self._seed)
        it2 = iterators.MultithreadIterator(dataset, 4, **self.options)
        it2.next()
        time.sleep(0.2)

        it2.reset()
        assert batch_a == it2.next()

        it2.next()
        it2.next()
        time.sleep(0.2)

        it2.reset()
        assert batch_b == it2.next()

    def test_random_state_serialize(self):
        dataset = self.RandomDataset()

        numpy.random.seed(self._seed)
        it1 = iterators.MultithreadIterator(dataset, 4, **self.options)
        it1.next()
        batch_a = it1.next()
        it1.next()
        it1.next()
        batch_b = it1.next()

        numpy.random.seed(self._seed)
        it2 = iterators.MultithreadIterator(dataset, 4, **self.options)
        it2.next()
        time.sleep(0.2)

        target = dict()
        it2.serialize(DummySerializer(target))
        assert batch_a == it2.next()
        it2 = iterators.MultithreadIterator(dataset, 4, **self.options)
        it2.serialize(DummyDeserializer(target))
        assert batch_a == it2.next()

        it2.next()
        it2.next()
        time.sleep(0.2)

        target = dict()
        it2.serialize(DummySerializer(target))
        assert batch_b == it2.next()
        it2 = iterators.MultithreadIterator(dataset, 4, **self.options)
        it2.serialize(DummyDeserializer(target))
        assert batch_b == it2.next()


testing.run_module(__name__, __file__)
