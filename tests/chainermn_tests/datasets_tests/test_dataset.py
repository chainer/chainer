from __future__ import with_statement
import unittest

from chainer import testing
import mpi4py.MPI
import numpy as np
import pytest

import chainermn
from chainermn.communicators.flat_communicator import FlatCommunicator
from chainermn.communicators.naive_communicator import NaiveCommunicator


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
