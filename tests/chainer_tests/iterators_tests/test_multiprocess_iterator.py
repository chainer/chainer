from __future__ import division
import copy
import errno
import os
import platform
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import numpy
import six

from chainer import iterators
from chainer import serializer
from chainer import testing
from chainer.testing import attr


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


class BaseTestMultiprocessIterator(object):

    def setUp(self):
        self.n_processes = 2
        self.options = {'n_processes': self.n_processes,
                        'n_prefetch': self.n_prefetch,
                        'shared_mem': self.shared_mem,
                        'maxtasksperchild': self.maxtasksperchild}
        if self.order_sampler is not None:
            self.options.update(
                {'order_sampler': self.order_sampler})

    @unittest.skipIf(platform.system() == 'Windows' and
                     int(platform.python_version_tuple()[0]) < 3,
                     'causes timeout in conda with Windows')
    def test_iterator_repeat(self):
        dataset = [1, 2, 3, 4, 5, 6]
        it = iterators.MultiprocessIterator(dataset, 2, **self.options)
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

    @unittest.skipIf(platform.system() == 'Windows' and
                     int(platform.python_version_tuple()[0]) < 3,
                     'causes timeout in conda with Windows')
    def test_iterator_list_type(self):
        dataset = [[i, numpy.zeros((10,)) + i] for i in range(6)]
        it = iterators.MultiprocessIterator(dataset, 2, **self.options)
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

    @unittest.skipIf(platform.system() == 'Windows' and
                     int(platform.python_version_tuple()[0]) < 3,
                     'causes timeout in conda with Windows')
    def test_iterator_tuple_type(self):
        dataset = [(i, numpy.zeros((10,)) + i) for i in range(6)]
        it = iterators.MultiprocessIterator(dataset, 2, **self.options)
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

    @unittest.skipIf(platform.system() == 'Windows' and
                     int(platform.python_version_tuple()[0]) < 3,
                     'causes timeout in conda with Windows')
    def test_iterator_dict_type(self):
        dataset = [{i: numpy.zeros((10,)) + i} for i in range(6)]
        it = iterators.MultiprocessIterator(dataset, 2, **self.options)
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

    @unittest.skipIf(platform.system() == 'Windows' and
                     int(platform.python_version_tuple()[0]) < 3,
                     'causes timeout in conda with Windows')
    def test_iterator_repeat_not_even(self):
        dataset = [1, 2, 3, 4, 5]
        it = iterators.MultiprocessIterator(dataset, 2, **self.options)

        batches = sum([it.next() for _ in range(5)], [])
        self.assertEqual(sorted(batches), sorted(dataset * 2))

    @unittest.skipIf(platform.system() == 'Windows' and
                     int(platform.python_version_tuple()[0]) < 3,
                     'causes timeout in conda with Windows')
    def test_iterator_not_repeat(self):
        dataset = [1, 2, 3, 4, 5]
        it = iterators.MultiprocessIterator(
            dataset, 2, repeat=False, **self.options)

        batches = sum([it.next() for _ in range(3)], [])
        self.assertEqual(sorted(batches), dataset)
        for _ in range(2):
            self.assertRaises(StopIteration, it.next)

    @unittest.skipIf(platform.system() == 'Windows' and
                     int(platform.python_version_tuple()[0]) < 3,
                     'causes timeout in conda with Windows')
    def test_iterator_not_repeat_not_even(self):
        dataset = [1, 2, 3, 4, 5]
        it = iterators.MultiprocessIterator(
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

    @unittest.skipIf(platform.system() == 'Windows' and
                     int(platform.python_version_tuple()[0]) < 3,
                     'causes timeout in conda with Windows')
    def test_iterator_shuffle_divisible(self):
        dataset = list(range(10))
        it = iterators.MultiprocessIterator(
            dataset, 10, **self.options)
        self.assertNotEqual(it.next(), it.next())

    @unittest.skipIf(platform.system() == 'Windows' and
                     int(platform.python_version_tuple()[0]) < 3,
                     'causes timeout in conda with Windows')
    def test_iterator_shuffle_nondivisible(self):
        dataset = list(range(10))
        it = iterators.MultiprocessIterator(
            dataset, 3, **self.options)
        out = sum([it.next() for _ in range(7)], [])
        self.assertNotEqual(out[0:10], out[10:20])

    @unittest.skipIf(platform.system() == 'Windows' and
                     int(platform.python_version_tuple()[0]) < 3,
                     'causes timeout in conda with Windows')
    def test_copy_not_repeat(self):
        dataset = [1, 2, 3, 4, 5]
        it = iterators.MultiprocessIterator(
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

    @unittest.skipIf(platform.system() == 'Windows' and
                     int(platform.python_version_tuple()[0]) < 3,
                     'causes timeout in conda with Windows')
    def test_reset(self):
        dataset = [1, 2, 3, 4, 5]
        it = iterators.MultiprocessIterator(
            dataset, 2, repeat=False, **self.options)

        for trial in range(4):
            batches = sum([it.next() for _ in range(3)], [])
            self.assertEqual(sorted(batches), dataset)
            for _ in range(2):
                self.assertRaises(StopIteration, it.next)
            it.reset()

    @unittest.skipIf(platform.system() == 'Windows' and
                     int(platform.python_version_tuple()[0]) < 3,
                     'causes timeout in conda with Windows')
    def test_reset_middle(self):
        dataset = [1, 2, 3, 4, 5]
        it = iterators.MultiprocessIterator(
            dataset, 2, repeat=False, **self.options)

        for trial in range(4):
            it.next()
            it.reset()
            batches = sum([it.next() for _ in range(3)], [])
            self.assertEqual(sorted(batches), dataset)
            for _ in range(2):
                self.assertRaises(StopIteration, it.next)
            it.reset()

    @unittest.skipIf(platform.system() == 'Windows' and
                     int(platform.python_version_tuple()[0]) < 3,
                     'causes timeout in conda with Windows')
    def test_reset_repeat(self):
        dataset = [1, 2, 3, 4]
        it = iterators.MultiprocessIterator(
            dataset, 2, repeat=True, **self.options)

        for trial in range(4):
            batches = sum([it.next() for _ in range(4)], [])
            self.assertEqual(sorted(batches), sorted(2 * dataset))
            it.reset()

    @unittest.skipIf(platform.system() == 'Windows' and
                     int(platform.python_version_tuple()[0]) < 3,
                     'causes timeout in conda with Windows')
    def test_unsupported_reset_finalized(self):
        dataset = [1, 2, 3, 4]
        it = iterators.MultiprocessIterator(
            dataset, 2, repeat=False, **self.options)
        it.next()
        it.next()
        it.finalize()
        self.assertRaises(NotImplementedError, it.reset)


@testing.parameterize(*testing.product({
    'n_prefetch': [1, 2],
    'shared_mem': [None, 1000000],
    'order_sampler': [
        None, lambda order, _: numpy.random.permutation(len(order))],
    'maxtasksperchild': [None],
}))
class TestMultiprocessIterator(
        BaseTestMultiprocessIterator, unittest.TestCase):
    pass


@testing.parameterize(*testing.product({
    'n_prefetch': [1, 2],
    'shared_mem': [None, 1000000],
    'order_sampler': [
        None, lambda order, _: numpy.random.permutation(len(order))],
    'maxtasksperchild': [1, 10],
}))
@attr.slow
class TestMultiprocessIteratorSlow(
        BaseTestMultiprocessIterator, unittest.TestCase):
    pass


@testing.parameterize(*testing.product({
    'n_prefetch': [1, 2],
    'shared_mem': [None, 1000000],
    'order_sampler': [
        None, lambda order, _: numpy.random.permutation(len(order))],
}))
class TestMultiprocessIteratorSerialize(unittest.TestCase):

    def setUp(self):
        self.n_processes = 2
        self.options = {'n_processes': self.n_processes,
                        'n_prefetch': self.n_prefetch,
                        'shared_mem': self.shared_mem}
        if self.order_sampler is not None:
            self.options.update(
                {'shuffle': None, 'order_sampler': self.order_sampler})

    def test_iterator_serialize(self):
        dataset = [1, 2, 3, 4, 5, 6]
        it = iterators.MultiprocessIterator(dataset, 2, **self.options)

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

        it = iterators.MultiprocessIterator(dataset, 2, **self.options)
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

    def test_iterator_serialize_backward_compat(self):
        dataset = [1, 2, 3, 4, 5, 6]
        it = iterators.MultiprocessIterator(dataset, 2, **self.options)

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
        # older version does not have previous_epoch_detail
        del target['previous_epoch_detail']

        it = iterators.MultiprocessIterator(dataset, 2, **self.options)
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
    'n_prefetch': [1, 2],
    'shared_mem': [None, 1000000],
}))
class TestMultiprocessIteratorOrderSamplerEpochSize(unittest.TestCase):

    def setUp(self):
        def order_sampler(order, cur_pos):
            return numpy.repeat(numpy.arange(3), 2)
        self.n_processes = 2
        self.options = {'n_processes': self.n_processes,
                        'n_prefetch': self.n_prefetch,
                        'shared_mem': self.shared_mem,
                        'shuffle': None,
                        'order_sampler': order_sampler}

    def test_iterator_repeat(self):
        dataset = [1, 2, 3]
        it = iterators.MultiprocessIterator(dataset, 2, **self.options)
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
            self.assertAlmostEqual(it.epoch_detail, i + 6 / 6)
            self.assertAlmostEqual(it.previous_epoch_detail, i + 4 / 6)

            self.assertEqual(
                sorted(batch1 + batch2 + batch3), [1, 1, 2, 2, 3, 3])


class _NoSameIndicesOrderSampler(object):

    def __init__(self, batchsize):
        self.n_call = 0

    def __call__(self, current_order, current_pos):
        # all batches contain unique indices
        remaining = current_order[current_pos:]
        first = numpy.setdiff1d(numpy.arange(len(current_order)), remaining)
        second = numpy.setdiff1d(numpy.arange(len(current_order)), first)
        return numpy.concatenate((first, second))


class TestNoSameIndicesOrderSampler(unittest.TestCase):

    def test_no_same_indices_order_sampler(self):
        dataset = [1, 2, 3, 4, 5, 6]
        batchsize = 5

        it = iterators.MultiprocessIterator(
            dataset, batchsize,
            order_sampler=_NoSameIndicesOrderSampler(batchsize))
        for _ in range(5):
            batch = it.next()
            self.assertEqual(len(numpy.unique(batch)), batchsize)


class _InvalidOrderSampler(object):

    def __init__(self):
        self.n_call = 0

    def __call__(self, _order, _):
        order = numpy.arange(len(_order) - self.n_call)
        self.n_call += 1
        return order


class TestMultiprocessIteratorInvalidOrderSampler(unittest.TestCase):

    def test_invalid_order_sampler(self):
        dataset = [1, 2, 3, 4, 5, 6]

        with self.assertRaises(ValueError):
            it = iterators.MultiprocessIterator(
                dataset, 6, shuffle=None,
                order_sampler=_InvalidOrderSampler())
            it.next()


class TestMultiprocessIteratorConcurrency(unittest.TestCase):

    def test_finalize_not_deadlock(self):
        dataset = numpy.ones((1000, 1000))
        it = iterators.MultiprocessIterator(dataset, 10, n_processes=4)
        for _ in range(10):
            it.next()

        t = threading.Thread(target=lambda: it.finalize())
        t.daemon = True
        t.start()
        t.join(5)
        deadlock = t.is_alive()

        self.assertFalse(deadlock)


class TestMultiprocessIteratorDeterminancy(unittest.TestCase):

    def setUp(self):
        self._seed = 3141592653
        self._random_bak = numpy.random.get_state()

    def tearDown(self):
        numpy.random.set_state(self._random_bak)

    def test_reproduce_same_permutation(self):
        dataset = [1, 2, 3, 4, 5, 6]
        order_sampler1 = iterators.ShuffleOrderSampler(
            numpy.random.RandomState(self._seed))
        it1 = iterators.MultiprocessIterator(
            dataset, 6, order_sampler=order_sampler1)
        order_sampler2 = iterators.ShuffleOrderSampler(
            numpy.random.RandomState(self._seed))
        it2 = iterators.MultiprocessIterator(
            dataset, 6, order_sampler=order_sampler2)
        for _ in range(5):
            self.assertEqual(it1.next(), it2.next())


@testing.parameterize(*testing.product({
    'n_prefetch': [1, 2],
    'shared_mem': [None, 1000000],
    'order_sampler': [
        None, lambda order, _: numpy.random.permutation(len(order))],
}))
class TestMultiprocessIteratorInterruption(unittest.TestCase):

    # unless you're debugging tests, this should be false
    show_interruption_msg = False

    def setUp(self):
        self.code_path = None
        if not self.show_interruption_msg:
            self.nullfd = os.open(os.devnull, os.O_WRONLY)

    def tearDown(self):
        if not self.show_interruption_msg:
            os.close(self.nullfd)
        if self.code_path is not None:
            os.remove(self.code_path)

    def run_code(self, dataset, n_processes, operation):
        code_template = """
import os
import random
import sys
import time
from chainer import iterators

# Using `multiprocessing` on Windows Python 2.7 requires
# that the script can be found on `sys.path`.
# See https://bugs.python.org/issue19946
sys.path.append(os.path.dirname(__file__))

class InfiniteWaitDataSet(object):
    def __len__(self):
        return 1000000
    def __getitem__(self, _):
        time.sleep(1000000)
infinite_wait = InfiniteWaitDataSet()

class NoWaitDataSet(object):
    def __len__(self):
        return 1000000
    def __getitem__(self, _):
        return 0
no_wait = NoWaitDataSet()

if __name__ == '__main__':
    if {shared_mem} is not None and {dataset} is infinite_wait:
        iterators.MultiprocessIterator._interruption_testing = True
    it = iterators.MultiprocessIterator({dataset}, 100,
                                        shuffle={shuffle},
                                        n_processes={n_processes},
                                        n_prefetch={n_prefetch},
                                        shared_mem={shared_mem},
                                        order_sampler={order_sampler})
    {operation}
        """
        code = code_template.format(dataset=dataset,
                                    shuffle=None,
                                    n_processes=n_processes,
                                    n_prefetch=self.n_prefetch,
                                    shared_mem=self.shared_mem,
                                    order_sampler=self.order_sampler,
                                    operation=operation)
        fd, self.code_path = tempfile.mkstemp(suffix='.py')
        os.write(fd, six.b(code))
        os.close(fd)

        if self.shared_mem is not None and dataset is 'infinite_wait':
            stdout = subprocess.PIPE
        else:
            stdout = None
        stderr = None if self.show_interruption_msg else self.nullfd
        self.p = subprocess.Popen([sys.executable, self.code_path],
                                  stdout=stdout, stderr=stderr)
        if stdout is None:
            self.child_pids = []
        else:
            self.child_pids = list(map(int, self.p.stdout.readline().split()))

    def send_sigint(self):
        # `signal.CTRL_C_EVENT` is also sent to the test process itself.
        # See https://docs.python.org/3.6/library/os.html#os.kill
        # So we need to wait the signal and ignore it.
        # We can NOT ignore the signal by modifying the signal handler here.
        # If we temporary ignores the signal, the signal will sent again
        # when the signal handler is restored.
        # If we ignore the signal permanently, we couldn't interrupt the test.
        if os.name == 'nt':
            try:
                os.kill(self.p.pid, signal.CTRL_C_EVENT)
                while True:
                    pass
            except KeyboardInterrupt:
                pass
        else:
            os.kill(self.p.pid, signal.SIGINT)

    def killall(self):
        # try waiting the root process
        # Python 2.7 doesn't have `subprocess.TimeoutExpired`,
        # so we couldn't use `p.wait(10)`.
        for _ in range(10):
            time.sleep(1)
            if self.p.poll() is not None:
                self.p.wait()
                break

        pids = [self.p.pid] + self.child_pids

        was_alive = False
        for pid in pids:
            try:
                if os.name == 'nt':
                    os.kill(pid, signal.SIGTERM)
                else:
                    os.kill(pid, signal.SIGKILL)
            except OSError as e:
                # no such pid (unix)
                if e.errno == errno.ESRCH:
                    pass
                # process terminated but its handle remains (Windows)
                elif e.errno == errno.EACCES:
                    pass
                # process terminated and its handle erased (Windows)
                elif e.errno == errno.EINVAL:
                    pass
                else:
                    raise
            else:  # process had existed and successfully killed
                was_alive = True
        return was_alive

    @unittest.skip
    def test_interrupt_infinite_wait_batch(self):
        # TODO(niboshi): See: https://github.com/chainer/chainer/issues/3383
        self.run_code(dataset='infinite_wait',
                      n_processes=2,
                      operation='it.next()')
        time.sleep(1.5)
        self.send_sigint()
        self.assertFalse(self.killall())

    @unittest.skip
    def test_interrupt_no_wait_batch(self):
        # TODO(niboshi): See: https://github.com/chainer/chainer/issues/3383
        self.run_code(dataset='no_wait',
                      n_processes=2,
                      operation='time.sleep(1000)')
        time.sleep(1.5)
        self.send_sigint()
        self.assertFalse(self.killall())


class StallingDataset(object):

    def __init__(self, nth, sleep):
        self.data = [0, 1, 2, 3, 4]
        self.nth = nth
        self.sleep = sleep

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        if i == self.nth:
            time.sleep(self.sleep)
        return self.data[i]


@testing.parameterize(*testing.product({
    'nth': [0, 1, 2],  # A fetch of whatth item will stall?
}))
class TestMultiprocessIteratorStalledDatasetDetection(unittest.TestCase):

    def test_stalled_getitem(self):
        nth = self.nth
        batch_size = 2
        sleep = 0.5
        timeout = 0.1

        dataset = StallingDataset(nth, sleep)
        it = iterators.MultiprocessIterator(
            dataset, batch_size=batch_size, shuffle=False,
            dataset_timeout=timeout, repeat=False)

        # TimeoutWarning should be issued.
        warning_cls = iterators.MultiprocessIterator.TimeoutWarning
        data = []
        # No warning until the stalling batch
        for i in range(nth // batch_size):
            data.append(it.next())
        # Warning on the stalling batch
        with testing.assert_warns(warning_cls):
            data.append(it.next())
        # Retrieve data until the end
        while True:
            try:
                data.append(it.next())
            except StopIteration:
                break

        # All data must be retrieved
        assert data == [
            dataset.data[i * batch_size: (i+1) * batch_size]
            for i in range((len(dataset) + batch_size - 1) // batch_size)]


testing.run_module(__name__, __file__)
