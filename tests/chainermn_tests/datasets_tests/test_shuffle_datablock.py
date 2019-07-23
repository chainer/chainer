import itertools
import pytest

import numpy as np
from numpy.testing import assert_array_equal

import chainermn
from chainermn.datasets.shuffle_datablock import _count_table
from chainermn.datasets.shuffle_datablock import _send_recv_pairs
from chainermn.datasets.shuffle_datablock import shuffle_data_chunks


@pytest.mark.parametrize('chunk_size,force_equal_length',
                         list(itertools.product([1000, 100000],
                                                [True, False])))
def test_shuffle_datablocks(chunk_size, force_equal_length):
    comm = chainermn.create_communicator('flat')


    # Rank i generates data = range(10**i)
    num_elem = 10 ** (comm.rank + 1)
    data = range(num_elem, num_elem * 2 + 3)

    total_data_size = sum(10 ** r for r in range(comm.size))

    data = shuffle_data_chunks(comm, data,
                               force_equal_length=force_equal_length,
                               chunk_size=chunk_size)

    if force_equal_length:
        length_all = comm.allgather(np.array([len(data)]))

        # (array([6]), array([6])) -> [6, 6]
        length_all = [xs[0] for xs in length_all]

        assert len(set(length_all)) == 1  # All rank have the same length

        # When force_equal_length is True, each data must be longer than the
        # average of len(data)
        avg_length = total_data_size * 1.0 / comm.size
        assert len(data) >= avg_length


def test_send_recv_pairs():
    # 3x3 table
    answer = [(0, 0), (0, 1), (0, 2),
              (1, 0), (1, 1), (1, 2),
              (2, 0), (2, 1), (2, 2)]
    assert_array_equal(_send_recv_pairs(3), answer)

    # 4x4 table
    answer = [(0, 0), (0, 1), (0, 2), (0, 3),
              (1, 0), (1, 1), (1, 2), (1, 3),
              (2, 0), (2, 1), (2, 2), (2, 3),
              (3, 0), (3, 1), (3, 2), (3, 3)]
    assert_array_equal(_send_recv_pairs(4), answer)


def test_count_table():
    """A unit test for an internal function"""

    length_all = [0, 0, 100]
    answer = np.array([[0,  0,  0],
                       [0,  0,  0],
                       [33, 33, 34]])
    assert_array_equal(_count_table(length_all), answer)

    length_all = [0, 0, 101]
    answer = np.array([[0, 0, 0],
                       [0, 0, 0],
                       [34, 33, 34]])
    assert_array_equal(_count_table(length_all), answer)

    length_all = [100]
    answer = np.array([[100]])
    assert_array_equal(_count_table(length_all), answer)

    length_all = [10, 20, 30]
    answer = np.array([[4, 3, 3],
                       [6, 7, 7],
                       [10, 10, 10]])
    assert_array_equal(_count_table(length_all), answer)
