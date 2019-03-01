import chainer
import chainer.testing
import chainer.testing.attr
import numpy
import pytest
import unittest

import chainermn
import chainermn.functions


class TestCollectiveCommunication(unittest.TestCase):

    def setup(self, gpu):
        numpy.random.seed(42)

        if gpu:
            self.communicator = chainermn.create_communicator('hierarchical')
            self.device = self.communicator.intra_rank
            chainer.cuda.get_device_from_id(self.device).use()
        else:
            self.communicator = chainermn.create_communicator('naive')

        if self.communicator.size < 2:
            pytest.skip('This test is for multinode')

    def check_all_gather(self, xs):
        x = xs[self.communicator.rank]
        ys = chainermn.functions.allgather(self.communicator, x)
        e = 0
        for i, y in enumerate(ys):
            e += chainer.functions.mean_squared_error(y, xs[i])
        e.backward()

        # Check backward does not fall in deadlock, and error = 0.
        self.assertEqual(e.data, 0)
        self.assertEqual(e.grad, 1)

    def test_all_gather_cpu(self):
        self.setup(False)
        xs = [chainer.Variable(
            numpy.random.normal(size=(10, i + 1)).astype(numpy.float32))
            for i in range(self.communicator.size)]
        self.check_all_gather(xs)

    @chainer.testing.attr.gpu
    def test_all_gather_gpu(self):
        self.setup(True)
        xs = [chainer.Variable(
            numpy.random.normal(size=(10, i + 1)).astype(numpy.float32))
            for i in range(self.communicator.size)]
        for x in xs:
            x.to_gpu()
        self.check_all_gather(xs)

    def check_all_to_all(self, xs):
        ys = chainermn.functions.alltoall(self.communicator, xs)

        y = chainer.functions.sum(ys[0])
        for _y in ys[1:]:
            y += chainer.functions.sum(_y)

        y.backward()

        # Check if gradients are passed back without deadlock.
        self.assertTrue(xs[0].grad is not None)

    def test_all_to_all_cpu(self):
        self.setup(False)
        data = [
            chainer.Variable(numpy.zeros(
                (self.communicator.rank, i), dtype=numpy.float32))
            for i in range(self.communicator.size)]
        self.check_all_to_all(data)

    @chainer.testing.attr.gpu
    def test_all_to_all_gpu(self):
        self.setup(True)

        chainer.cuda.get_device_from_id(self.device).use()
        data = [
            chainer.Variable(numpy.zeros(
                (self.communicator.rank + 1, i + 1), dtype=numpy.float32))
            for i in range(self.communicator.size)]
        for x in data:
            x.to_gpu()
        self.check_all_to_all(data)

    def check_bcast(self, x):
        root = 0
        if self.communicator.rank == root:
            y = chainermn.functions.bcast(
                self.communicator, x, root)
        else:
            y = chainermn.functions.bcast(
                self.communicator, None, root)
        e = chainer.functions.mean_squared_error(y, x)
        e.backward()

        # Check backward does not fall in deadlock, and error = 0 in root.
        if self.communicator.rank == root:
            self.assertEqual(e.data, 0)
            self.assertEqual(e.grad, 1)

    def test_bcast_cpu(self):
        self.setup(False)
        x = chainer.Variable(
            numpy.random.normal(size=(100, 100)).astype(numpy.float32))
        self.check_bcast(x)

    @chainer.testing.attr.gpu
    def test_bcast_gpu(self):
        self.setup(True)
        x = chainer.Variable(
            numpy.random.normal(size=(100, 100)).astype(numpy.float32))
        x.to_gpu()
        self.check_bcast(x)

    def check_gather(self, xs):
        root = 0
        # All processes receive the same xs since seed is fixed.
        x = xs[self.communicator.rank]

        if self.communicator.rank == root:
            ys = chainermn.functions.gather(
                self.communicator, x, root)
            e = 0
            for i, y in enumerate(ys):
                e += chainer.functions.mean_squared_error(y, xs[i])
            e.backward()

            # Check backward does not fall in deadlock, and error = 0.
            self.assertEqual(e.data, 0)
            self.assertEqual(e.grad, 1)

        else:
            phi = chainermn.functions.gather(
                self.communicator, x, root)
            phi.backward()

            # Check backward does not fall in deadlock.
            self.assertTrue(x.grad is not None)

    def test_gather_cpu(self):
        self.setup(False)
        xs = [chainer.Variable(
            numpy.random.normal(size=(100, 100)).astype(numpy.float32))
            for _ in range(self.communicator.size)]
        self.check_gather(xs)

    @chainer.testing.attr.gpu
    def test_gather_gpu(self):
        self.setup(True)
        xs = [chainer.Variable(
            numpy.random.normal(size=(100, 100)).astype(numpy.float32))
            for _ in range(self.communicator.size)]
        for x in xs:
            x.to_gpu()
        self.check_gather(xs)

    def test_gatherv_cpu(self):
        self.setup(False)
        xs = [chainer.Variable(
            numpy.random.normal(size=(i + 1, i + 1)).astype(numpy.float32))
            for i in range(self.communicator.size)]
        self.check_gather(xs)

    @chainer.testing.attr.gpu
    def test_gatherv_gpu(self):
        self.setup(True)
        xs = [chainer.Variable(
            numpy.random.normal(size=(i + 1, i + 1)).astype(numpy.float32))
            for i in range(self.communicator.size)]
        for x in xs:
            x.to_gpu()
        self.check_gather(xs)

    def check_scatter(self, xs):
        # All processes receive the same xs since seed is fixed.
        root = 0

        y = chainermn.functions.scatter(
            self.communicator,
            xs if self.communicator.rank == root else None,
            root)
        x = xs[self.communicator.rank]
        e = chainer.functions.mean_squared_error(y, x)
        e.backward()

        # Check backward does not fall in deadlock, and error = 0.
        self.assertEqual(e.data, 0)
        self.assertEqual(e.grad, 1)

    def test_scatter_cpu(self):
        self.setup(False)
        xs = [chainer.Variable(
            numpy.random.normal(size=(100, 100)).astype(numpy.float32))
            for _ in range(self.communicator.size)]
        self.check_scatter(xs)

    @chainer.testing.attr.gpu
    def test_scatter_gpu(self):
        self.setup(True)
        xs = [chainer.Variable(
            numpy.random.normal(size=(100, 100)).astype(numpy.float32))
            for _ in range(self.communicator.size)]
        for x in xs:
            x.to_gpu()
        self.check_scatter(xs)

    def test_scatterv_cpu(self):
        self.setup(False)
        xs = [chainer.Variable(
            numpy.random.normal(size=(i + 1, i + 1)).astype(numpy.float32))
            for i in range(self.communicator.size)]
        self.check_scatter(xs)

    @chainer.testing.attr.gpu
    def test_scatterv_gpu(self):
        self.setup(True)
        xs = [chainer.Variable(
            numpy.random.normal(size=(i + 1, i + 1)).astype(numpy.float32))
            for i in range(self.communicator.size)]
        for x in xs:
            x.to_gpu()
        self.check_scatter(xs)
