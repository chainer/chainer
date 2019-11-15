from __future__ import with_statement
import itertools
import unittest

import mpi4py.MPI
import numpy as np
import pytest

from chainer import testing
import chainermn
from chainermn.communicators.flat_communicator import FlatCommunicator
from chainermn.communicators.naive_communicator import NaiveCommunicator
import chainerx as chx


class TestDataset(unittest.TestCase):

    def setUp(self):
        self.mpi_comm = mpi4py.MPI.COMM_WORLD
        self.communicator = NaiveCommunicator(self.mpi_comm)

    def check_scatter_dataset(self, original_dataset, shuffle=False, root=0):
        if self.communicator.rank != root:
            original_dataset = None
        my_dataset = chainermn.scatter_dataset(
            original_dataset, self.communicator,
            shuffle=shuffle, root=root)
        sub_datasets = self.communicator.gather_obj(my_dataset, root=root)

        if self.communicator.rank == root:
            # Test the sizes
            sub_sizes = [len(sub_dataset) for sub_dataset in sub_datasets]
            self.assertEqual(len(set(sub_sizes)), 1)
            sub_size = sub_sizes[0]
            self.assertLessEqual(
                len(original_dataset), sub_size * self.mpi_comm.size)
            self.assertGreater(
                len(original_dataset), (sub_size - 1) * self.mpi_comm.size)

            # Test the content of scattered datasets
            joined_dataset = sum((sub_dataset[:]
                                  for sub_dataset in sub_datasets), [])

            # NOTE: The values in `original_dataset` and
            # `joined_dataset` must be casted to int to compare.
            # There are 2 backgrounds on this issue.
            #
            # (1) numpy and cupy/chainerx have different behaviours on
            # 1-element array. Numpy implicitly converts a 1-element array to
            # a scalar value.
            # type(numpy.array([1])[0])
            # =>  <class 'numpy.int64'>  # Scalar
            # type(chainerx.array([1])[0])
            # => <class 'chainerx.ndarray'>  # array of one element
            #
            # (2) Two different ChainerX arrays are never identical in the
            # context of `set()`.
            # set([chainerx.array([0]), chainerx.array([0])])
            # => {array([0], shape=(1,), dtype=int64, device='native:0'),
            #     array([0], shape=(1,), dtype=int64, device='native:0')}

            joined_dataset = [int(e) for e in joined_dataset]
            original_dataset = [int(e) for e in original_dataset]
            self.assertEqual(set(joined_dataset), set(original_dataset))

    def test_scatter_dataset(self):
        n = self.communicator.size

        for shuffle in [True, False]:
            for root in range(self.communicator.size):
                self.check_scatter_dataset([], shuffle, root)
                self.check_scatter_dataset([0], shuffle, root)
                self.check_scatter_dataset(list(range(n)), shuffle, root)
                self.check_scatter_dataset(list(range(n * 5 - 1)),
                                           shuffle, root)

                self.check_scatter_dataset(np.array([]), shuffle, root)
                self.check_scatter_dataset(np.array([0]), shuffle, root)
                self.check_scatter_dataset(np.arange(n), shuffle, root)
                self.check_scatter_dataset(np.arange(n * 5 - 1), shuffle, root)

                self.check_scatter_dataset(chx.array([]), shuffle, root)
                self.check_scatter_dataset(chx.array([0]), shuffle, root)
                self.check_scatter_dataset(chx.arange(n), shuffle, root)
                self.check_scatter_dataset(
                    chx.arange(n * 5 - 1), shuffle, root)


def scatter_large_data(communicator):
    data = []
    if communicator.rank == 0:
        data = ['test'] * 2000000000
    data = chainermn.scatter_dataset(data, communicator)
    assert len(data) > 0


@testing.attr.slow
def test_scatter_large_dataset_naive():
    mpi_comm = mpi4py.MPI.COMM_WORLD
    communicator = NaiveCommunicator(mpi_comm)

    # This test only runs when comm.size >= 2.
    if communicator.size == 1:
        pytest.skip('This test is for multinode')

    scatter_large_data(communicator)


@testing.attr.gpu
@testing.attr.slow
def test_scatter_large_dataset_flat():
    mpi_comm = mpi4py.MPI.COMM_WORLD
    communicator = FlatCommunicator(mpi_comm)

    # This test only runs when comm.size >= 2.
    if communicator.size == 1:
        pytest.skip('This test is for multinode')

    scatter_large_data(communicator)


def test_scatter_index_one():
    it = chainermn.datasets.scatter._scatter_index(10, 3, False)
    split = [(0, 0, 4), (1, 4, 7), (2, 7, 10)]
    for lhs, rhs in zip(split, it):
        assert lhs == rhs


@pytest.mark.parametrize('combination', [
    [10, 3], [1244, 23], [2, 1], [230945, 237]])
def test_scatter_index(combination):
    length, size = combination
    it = chainermn.datasets.scatter._scatter_index(length, size, False)
    union = set()
    total = []
    subsets = []
    for (_, b, e) in it:
        subset = list(range(b, e))
        subsets.append(subset)
        total.extend(subset)
        for x in subset:
            union.add(x)
    assert length == len(total)  # no duplication
    assert length == len(union)  # no duplication & no lacking
    for lhs, rhs in itertools.combinations(subsets, 2):
        set(lhs).isdisjoint(set(rhs))
        assert abs(len(lhs) - len(rhs)) <= 1
