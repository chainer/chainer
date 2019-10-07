import functools
import itertools
import pytest

import numpy as np
from numpy.testing import assert_array_equal

import chainermn
from chainermn.datasets.shuffle_datablock import _calc_alltoall_send_counts
from chainermn.datasets.shuffle_datablock import shuffle_data_blocks


def _numpy_flatten1(ary):
    """1-level flatten"""
    return functools.reduce(lambda a, b: np.concatenate([a,b], axis=0), ary)


def _gather_check(comm, orig_data, local_data, root=0):
    # Check if gather()-ed data is equivalent to the original data
    data_gathered = comm.mpi_comm.gather(np.array(local_data), root=0)
    if comm.rank == root:
        data_gathered = _numpy_flatten1(data_gathered)
        assert sorted(data_gathered) == sorted(orig_data)


@pytest.mark.parametrize('block_size,force_equal_length',
                         list(itertools.product([1, 1000],
                                                [False, True])))
def test_shuffle_datablocks(block_size, force_equal_length):
    comm = chainermn.create_communicator('naive')

    # Rank r generates data of length 10**min(r,3), to achieve
    # good balance of test coverage and test speed
    num_elem = 10 ** min(comm.rank, 3)
    data = range(num_elem, num_elem * 2)
    assert len(data) == num_elem

    data_all = []
    for i in range(comm.size):
        num_elem = 10 ** min(i, 3)
        data_all += list(range(num_elem, num_elem * 2))

    total_data_size = sum(10 ** min(r, 3) for r in range(comm.size))

    data = shuffle_data_blocks(comm, data,
                               force_equal_length=force_equal_length,
                               block_size=block_size)

    # (array([6]), array([6])) -> [6, 6]
    length_all = [x[0] for x in comm.allgather(np.array([len(data)]))]

    if force_equal_length:
        assert sum(length_all) >= total_data_size
        assert len(set(length_all)) == 1  # All ranks have the same length
    else:
        assert sum(length_all) == total_data_size
        assert max(length_all) - min(length_all) <= 1
        _gather_check(comm, data_all, data)


@pytest.mark.parametrize('length,force_equal_length',
                         list(itertools.product(
                             [10, 100, 1000],
                             [False, True])))
def test_shuffle_datablocks_scatter(length, force_equal_length):
    # shuffle_datablocks()'s functionality is a superset of scatter_dataset
    # if a single rank generates all data and scatter them.
    comm = chainermn.create_communicator('naive')
    if comm.rank == 0:
        data = range(length)
    else:
        data = []

    data = shuffle_data_blocks(comm, data, force_equal_length, block_size=10)

    if force_equal_length:
        assert len(data) == (length - 1) // comm.size + 1
    else:
        rem = length % comm.size
        if comm.rank < rem:
            assert len(data) == length // comm.size + 1
        else:
            assert len(data) == length // comm.size

    if not force_equal_length:
        _gather_check(comm, range(length), data)
