import unittest

import numpy
import six

from chainer import datasets
from chainer import testing


@testing.parameterize(*testing.product({'n_procs': list(range(1, 8))}))
class TestMultiprocessLoader(unittest.TestCase):

    def setUp(self):
        self.array = numpy.arange(10)
        self.baseset = datasets.SimpleDataset('baseset', self.array)

    def test_name(self):
        loader = datasets.MultiprocessLoader(self.baseset)
        self.assertEqual(loader.name, self.baseset.name)

    def test_len(self):
        loader = datasets.MultiprocessLoader(self.baseset)
        self.assertEqual(len(loader), len(self.baseset))

    def test_getitem(self):
        loader = datasets.MultiprocessLoader(self.baseset)
        for x, y in zip(loader, self.baseset):
            self.assertEqual(x, y)

    def test_iteration(self):
        loader = datasets.MultiprocessLoader(self.baseset, self.n_procs)
        it = loader.get_batch_iterator(repeat=False, auto_shuffle=False)
        result = list(it)
        it.finalize()

        for x, y in zip(self.array, result):
            self.assertEqual(len(y), 1)
            self.assertEqual(x, y[0])

    def test_batch_iteration(self):
        loader = datasets.MultiprocessLoader(self.baseset, self.n_procs)
        it = loader.get_batch_iterator(batchsize=3, repeat=False,
                                       auto_shuffle=False)
        n_batch = (len(self.array) - 1) // 3 + 1

        result = []
        for i in range(n_batch):
            result += list(it.next())
        with self.assertRaises(StopIteration):
            it.next()

        self.assertEqual(set(result), set(self.array))

    def test_batch_iteration_repeat(self):
        loader = datasets.MultiprocessLoader(self.baseset, self.n_procs)
        it = loader.get_batch_iterator(batchsize=3, auto_shuffle=False)

        for i in six.moves.range(20):
            self.assertEqual(it.epoch, i * 3 // len(self.array))
            x = set((i * 3 + j) % len(self.array) for j in range(3))
            y = it.next()
            self.assertEqual(set(y), x)
