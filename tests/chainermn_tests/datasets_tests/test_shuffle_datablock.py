import itertools
import pytest

import numpy as np
from numpy.testing import assert_array_equal

import chainermn
from chainermn.datasets.shuffle_datablock import _calc_alltoall_send_counts
from chainermn.datasets.shuffle_datablock import shuffle_data_blocks


@pytest.mark.parametrize('block_size,force_equal_length',
                         list(itertools.product([1000, 100000],
                                                [True, False])))
def test_shuffle_datablocks(block_size, force_equal_length):
    comm = chainermn.create_communicator('naive')

    # Rank r generates data of length 10**r
    num_elem = 10 ** comm.rank
    data = range(num_elem, num_elem * 2)
    assert len(data) == num_elem

    print("len(data) = {}".format(len(data)))

    total_data_size = sum(10 ** r for r in range(comm.size))

    data = shuffle_data_blocks(comm, data,
                               force_equal_length=force_equal_length,
                               block_size=block_size)

    # (array([6]), array([6])) -> [6, 6]
    length_all = [x[0] for x in comm.allgather(np.array([len(data)]))]

    if force_equal_length:
        assert len(set(length_all)) == 1  # All ranks have the same length
    else:
        assert max(length_all) - min(length_all) <= 1
