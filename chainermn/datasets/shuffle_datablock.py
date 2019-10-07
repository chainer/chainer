import itertools
import random

import chainer
import numpy
import functools


def _increment_send_counts(table, send_rank, recv_rank, n=1):
    key = (send_rank, recv_rank)
    table.setdefault(key, 0)
    table[key] += n


def _flatten1(ary):
    """1-level flatten"""
    return functools.reduce(lambda a, b: a + b, ary)


def _calc_new_length(cur_length_all, block_length_all):
    # the total number of elements over all processes, including
    # already-scattered data.
    size = len(cur_length_all)
    total = sum(cur_length_all) + sum(block_length_all)

    # new_length: new number of elements of all processes after shuffling
    if total % size == 0:
        new_length_all = [total // size] * size
    else:
        rem = total % size
        rem_all = [1 if i < rem else 0 for i in range(size)]
        new_length_all = [total // size + rem_all[i] for i in range(size)]

    return new_length_all


def _calc_alltoall_send_counts(cur_length_all, block_length_all):
    """
    Calculate send counts table for all_to_all() communication from current
    data length and loaded block length.

    For example, if length of loaded blocks of all processes are
       [1, 10, 18]
    then we will perform MPI_alltoallv() communication so that lengths are
       [10, 10, 9]   (force_equal_length is False)
    or
       [10, 10, 10]  (force_equal_length is True)
    because total number of elements are 29.

    In this case, if `force_equal_length` is False,
    the communication matrix, namely send_counts, is like:
       2 --> 0  (9 elements)
    so
       send_counts = {(2, 0): 9}

    :param cur_length_all: Already-loaded data length
    :param block_length_all: Length of the newly loaded block
    :param force_equal_length: If force equal length
    :return: A dict of send counts
    """
    size = len(cur_length_all)

    new_length_all = _calc_new_length(cur_length_all, block_length_all)

    # If diff_length[rank] is >0, then the rank has more elements than
    # expected, so it distribute the extra data to other ranks.
    diff_length = [block_length_all[i] + cur_length_all[i] - new_length_all[i]
                   for i in range(size)]

    send_counts = {}  # send_count table

    # Need to calculate the number of elements sent <self->self>
    # for MPI_Alltoallv().
    for rank in range(size):
        if diff_length[rank] > 0:
            # If diff_length[rank] > 0, the rank sends data to other process
            # as well as to itself.
            send_counts[(rank, rank)] = \
                block_length_all[rank] - diff_length[rank]
        else:
            # Otherwise, the rank sents not data and
            # just keeps the block for itself
            send_counts[(rank, rank)] = block_length_all[rank]

    # calculate the all-to-all send_counts as a dict.
    # NOTE: mpi4py's alltoall() is similar to MPI_Alltoallv().
    #       mpi4py does not have alltoallv().
    # dst_rank and src_rank go through diff_length array once so O(N)
    dst_rank = 0
    for src_rank in range(size):
        while diff_length[src_rank] > 0:
            # src_rank has elements to send to other ranks
            # find receiver(s)
            while diff_length[dst_rank] >= 0:
                dst_rank = (dst_rank + 1) % size
            send_cnt = diff_length[src_rank]
            recv_cnt = diff_length[dst_rank]
            if send_cnt <= -recv_cnt:
                # dst_rank can recieve more elements than
                # src_rank can send. Thus src_rank's send_cnt becomes 0
                send_counts[(src_rank, dst_rank)] = send_cnt
                diff_length[src_rank] = 0
                diff_length[dst_rank] += send_cnt
            else:
                # dst_rank can receive less elements than
                # src_rank can send. Thus dst_rank's recv_cnt becomes 0
                # NOTE: recv_cnt is a <0 value
                send_counts[(src_rank, dst_rank)] = -recv_cnt
                diff_length[src_rank] += recv_cnt
                diff_length[dst_rank] = 0
            if diff_length[src_rank] == 0:
                break

    # All the diffs are resolved by communication.
    assert all(d == 0 for d in diff_length)
    return send_counts, new_length_all


def _exchange_block(comm, data, block, cur_length_all, block_length_all):
    send_counts, new_length_all = _calc_alltoall_send_counts(
        cur_length_all, block_length_all)

    # Basically, send data is selected from the newly-loaded `block`,
    # but in some cases `block` is an empty array
    # i.e. when adjusting for force_equal_length
    if len(block) == 0:
        local_data = data
    else:
        local_data = block

    offset = 0
    send_buf = [[]] * comm.size
    for dest_rank in range(comm.size):
        num_elem = send_counts.get((comm.rank, dest_rank), 0)
        if num_elem == 0:
            continue

        send_buf[dest_rank] = local_data[offset:offset + num_elem]
        while len(send_buf[dest_rank]) < num_elem:
            # In case of force_equal_length, the process may have to send
            # more data than `block` has. We need to duplicate some elements
            # from `block`.
            send_buf[dest_rank].append(random.choice(block))
        offset = (offset + num_elem) % len(local_data)
    data += _flatten1(comm.mpi_comm.alltoall(send_buf))
    return new_length_all


def shuffle_data_blocks(comm, blocks, block_size, force_equal_length=True):
    """Exchange unbalanced blocks of data between all processes

    This function is useful when `scatter_dataset` is not suitable.
    For instance, the data is huge or the total length is unknown.

    shuffle_datablocks() function works in a iterative way.
    In a single iteartion, `block_size` items are loaded from `data_blocks`
    data source and exchanged between distributed processes so the number of
    elements in each process is roughly equal.

    If `force_equal_length` is True, some elements are duplicated when necessary
    so all processes have exactly the same number of elements after all
    iterations.

    :param comm: :class: `~chainermn.communicators.MpiCommunicatorBase`
    :param blocks: An iterative object to read data blocks.
    :param force_equal_length: Whether data length of each process is adjusted
                               by copying some elements, so that all processes
                               have exactly the same length of data.
                               This is required in training for correct
                               iteration/epoch counting.
                               In evaluation, however, the option can be False
                               if you don't want duplicated elements.
    :param block_size: Number of elements read from `data_blocks`
                       in an iteration
    :return: Shuffled data (in a list)
    """
    if not hasattr(comm, 'mpi_comm'):
        raise NotImplementedError('shuffle_data_blocks() function depends on'
                                  'MPI-based ChainerMN communicator.')

    chainer.utils.experimental(
        'chainermn.datasets.shuffle_datablock.shuffle_datablocks'
    )

    blocks = iter(blocks)
    data = []
    data_length_all = [0] * comm.size  # all processes start from data=[]

    # repeat until all processes consume all data
    while True:
        # Read a block of data; we need to use `itertools.islice`
        # to support both of list-like objects and generators
        block = list(itertools.islice(blocks, block_size))

        # wrap the length by numpy array to communicate via MPI
        if block is None:
            block_length = numpy.array([0])
        else:
            block_length = numpy.array([len(block)])

        # How many elements does each process have?
        block_length_all = [x[0] for x in comm.allgather(block_length)]

        # If nobody has any more data to send. done.
        if all(n == 0 for n in block_length_all):
            break
        else:
            data_length_all = _exchange_block(
                comm, data, block, data_length_all, block_length_all)

    if force_equal_length:
        max_data_len = max(data_length_all)
        min_data_len = min(data_length_all)
        if max_data_len != min_data_len:
            while len(data) < max_data_len:
                diff = max_data_len - len(data)
                data += data[:diff]  # NOTE: `data` may be shorter than `diff`

    return data
