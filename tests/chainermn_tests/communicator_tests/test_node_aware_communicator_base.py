import collections
import os
import unittest

import mpi4py.MPI
import pytest

from chainermn.communicators.mpi_communicator_base import MpiCommunicatorBase


class NodeAwareNaiveCommunicator(MpiCommunicatorBase):
    def __init__(self, mpi_comm):
        super(NodeAwareNaiveCommunicator, self).__init__(mpi_comm)

    def allreduce_grad(self, model):
        raise NotImplementedError()


class TestMpiCommunicatorBase(unittest.TestCase):

    def setUp(self):
        self.mpi_comm = mpi4py.MPI.COMM_WORLD
        self.communicator = NodeAwareNaiveCommunicator(self.mpi_comm)

    def test_intra_rank_with_env(self):
        if 'MV2_COMM_WORLD_LOCAL_RANK' in os.environ:  # MVAPICH
            expected = int(os.environ['MV2_COMM_WORLD_LOCAL_RANK'])
        elif 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:  # OpenMPI
            expected = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        else:
            pytest.skip('No MPI specified')

        self.assertEqual(self.communicator.intra_rank, expected)

    def test_intra_size_with_env(self):
        if 'MV2_COMM_WORLD_LOCAL_SIZE' in os.environ:  # MVAPICH
            expected = int(os.environ['MV2_COMM_WORLD_LOCAL_RANK'])
        elif 'OMPI_COMM_WORLD_LOCAL_SIZE' in os.environ:  # OpenMPI
            expected = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        else:
            pytest.skip('No MPI specified')

        self.assertEqual(self.communicator.intra_rank, expected)

    def test_inter_rank_and_size(self):
        ranks_and_sizes = self.mpi_comm.gather((
            self.communicator.inter_rank, self.communicator.inter_size))

        if self.mpi_comm.rank == 0:
            for inter_rank, inter_size in ranks_and_sizes:
                self.assertTrue(0 <= inter_rank < inter_size)

            sizes = list(set(x[1] for x in ranks_and_sizes))
            self.assertEqual(len(sizes), 1)
            size = sizes[0]

            ranks = list(sorted(set(x[0] for x in ranks_and_sizes)))
            self.assertEqual(ranks, list(range(size)))

    def test_intra_rank_and_size(self):
        ranks_and_sizes = self.mpi_comm.gather((
            self.communicator.intra_rank, self.communicator.intra_size,
            self.communicator.inter_rank, self.communicator.inter_size))

        if self.mpi_comm.rank == 0:
            for intra_rank, intra_size, _, _ in ranks_and_sizes:
                self.assertTrue(0 <= intra_rank < intra_size)

            inter_rank_to_intra_ranks = collections.defaultdict(list)

            for intra_rank, _, inter_rank, _ in ranks_and_sizes:
                inter_rank_to_intra_ranks[inter_rank].append(intra_rank)

            for ranks in inter_rank_to_intra_ranks.values():
                ranks.sort()

            for _, intra_size, inter_rank, _ in ranks_and_sizes:
                self.assertEqual(
                    inter_rank_to_intra_ranks[inter_rank],
                    list(range(intra_size)))
