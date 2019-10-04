import itertools
import random

import numpy
import functools


def _increment_send_counts(table, send_rank, recv_rank, n=1):
    key = (send_rank, recv_rank)
    table.setdefault(key, 0)
    table[key] += n


def _flatten1(ary):
    """1-level flatten"""
    return functools.reduce(lambda a, b: a + b, ary)


def _calc_alltoall_send_counts(cur_length_all, block_length_all,
                               force_equal_length):
    """
    Calculate send counts table for all_to_all() communication from current
    data length and loaded block length.

    :param cur_length_all: Already-loaded data length
    :param block_length_all: Length of the newly loaded block
    :param force_equal_length: If force equal length
    :return: A dict of send counts
    """
    print("==============")
    assert len(cur_length_all) == len(block_length_all)
    size = len(cur_length_all)

    # This condition is kept if we repeatedly use calc_all_to_all_send_counts()
    assert max(cur_length_all) - min(cur_length_all) <= 1

    # the total number of elements over all processes, including
    # already-scattered data.
    total = sum(cur_length_all) + sum(block_length_all)

    # new_length: new number of elements of all processes after shuffling
    if total % size == 0:
        new_length_all = [total // size] * size
    else:
        rem = total % size
        rem_all = [1 if i < rem else 0 for i in range(size)]
        new_length_all = [total // size + rem_all[i] for i in range(size)]

    assert sum(new_length_all) == total

    # If diff_length[rank] is >0, then the rank has more elements than
    # expected (indicated by new_length[rank]), so it has to send some elements
    # to other ranks
    diff_length = [block_length_all[i] + cur_length_all[i] - new_length_all[i]
                   for i in range(size)]

    # send_counts[(sender, recver)] = N
    send_counts = {}
    print("new_length_all = {}".format(new_length_all))
    print("diff_length = {}".format(diff_length))
    assert sum(diff_length) == 0

    # calculate the all-to-all send_counts as a dict.
    # NOTE: mpi4py's alltoall() is similar to MPI_Alltoallv().
    #       mpi4py does not have alltoallv().
    for send_rank in range(size):
        if diff_length[send_rank] > 0:
            # from_rank has `send_cnt` elements to send to other ranks
            # find receiver(s)
            for _recv_rank in range(size):
                recv_rank = (_recv_rank + size) % size
                send_cnt = diff_length[send_rank]
                recv_cnt = diff_length[recv_rank]
                if recv_cnt < 0:  # recv_rank is ready to receive some elems.
                    if send_cnt <= -recv_cnt:
                        send_counts[(send_rank, recv_rank)] = send_cnt
                        print("{} -> {} ({})".format(send_rank, recv_rank,
                                                     send_cnt))
                        diff_length[recv_rank] += send_cnt
                        diff_length[send_rank] = 0
                    else:
                        send_counts[(send_rank, recv_rank)] = -recv_cnt
                        print("{} -> {} ({})".format(send_rank, recv_rank,
                                                     -recv_cnt))
                        diff_length[send_rank] += recv_cnt
                        diff_length[recv_rank] = 0
                    if diff_length[send_rank] == 0:
                        break
    assert all(e == 0 for e in diff_length)

    # Need to calculate the number of elements sent <self->self>
    # for MPI_Alltoallv().
    num_sent_elems = [sum(send_counts.get((src_rank, dest_rank), 0)
                         for dest_rank in range(size))
                      for src_rank in range(size)]
    print("num_sent_elems = {}".format(num_sent_elems))

    for rank in range(size):
        send_counts[(rank, rank)]\
            = block_length_all[rank] - num_sent_elems[rank]

    if force_equal_length:
        print("_calc_alltoall_send_counts(): force_equal_length")
        # Communicate a few extra elements to force equal length.
        len_max = max(new_length_all)  # Adjust all ranks to the maximum length.

        # diff_length should be a list of 0 or -1.
        # "-1" rank should receive one element from another rank.
        diff_length = [ln - len_max for ln in new_length_all]
        assert set(diff_length) == {0, -1}

        send_rank = 0
        for recv_rank in range(size):
            if diff_length[recv_rank] == -1:
                # Find sender
                # for communication efficiency, it is prefarrable to receive
                # elements from nearby ranks and communication should be load-balanced.
                send_rank += 1
                while diff_length[send_rank] != 0:
                    send_rank = (send_rank + 1) % size
                print("_calc_alltoall_send_counts(): {} -> {} ({})".format(send_rank, recv_rank, 1))
                _increment_send_counts(send_counts, send_rank, recv_rank)
                diff_length[recv_rank] = 0
                new_length_all[recv_rank] += 1
        assert all(e == 0 for e in diff_length)
        print("_calc_alltoall_send_counts(): new_length = {}".format(new_length_all))
        print("_calc_alltoall_send_counts(): send_counts = {}".format(send_counts))
        assert len(set(new_length_all)) == 1

    return send_counts, new_length_all


def _exchange_block(comm, data, block, cur_length_all, block_length_all,
                    force_equal_counts):
    send_counts, new_length_all = _calc_alltoall_send_counts(
        cur_length_all, block_length_all, force_equal_counts)

    # Basically, send data is selected from the newly-loaded `block`,
    # but in some cases `block` is an empty array
    # i.e. when adjusting for force_equal_length
    if len(block) == 0:
        local_data = data
    else:
        local_data = block

    offset = 0
    send_buf = [[]] * comm.size
    print("_exchange_block(): send_counts = {}".format(send_counts))
    for dest_rank in range(comm.size):
        num_elem = send_counts.get((comm.rank, dest_rank), 0)
        print("_exchange_block(): ->{} num_elem = {}".format(dest_rank, num_elem))
        if num_elem == 0:
            continue

        send_buf[dest_rank] = local_data[offset:offset + num_elem]
        while len(send_buf[dest_rank]) < num_elem:
            # In case of force_equal_length, the process may have to send
            # more data than `block` has. We need to duplicate some elements
            # from `block`.
            send_buf[dest_rank].append(random.choice(block))
        offset = (offset + num_elem) % len(local_data)
    print("_exchange_block(): send_buf = {}".format(send_buf))
    assert len(send_buf) == comm.size
    print("_exchange_block(): data = {}".format(data))
    data += _flatten1(comm.mpi_comm.alltoall(send_buf))
    print("_exchange_block(): data = {}".format(data))
    return new_length_all


def shuffle_data_blocks(comm, data_blocks, force_equal_length=True,
                        block_size=10000):
    """Exchange unbalanced blocks of data between all processes

    This function is useful when `scatter_dataset` is not suitable.
    For instance, the data is huge or the total length is unknown.

    :param comm: ChainerMN communicator
    :param data_blocks: Sequence or generator to read blocks.
    :param force_equal_length: Whether data length of each process is adjusted
                               by copying some elements, so that all processes
                               have exactly the same length of data.
                               This is required in training for correct
                               iteration/epoch counting.
                               In evaluation, however, the option can be False
                               if you don't want duplicated elements.
    :param block_size: Number of elements read from `data_blocks` at a time
    :return: Shuffled data (in a list)
    """
    if not hasattr(comm, 'mpi_comm'):
        raise NotImplementedError('shuffle_data_blocks() function depends on'
                                  'MPI-based ChainerMN communicator.')

    if force_equal_length not in [None, True, False]:
        raise ValueError('Wrong value for `force_equal_length`:'
                         ' {}'.format(force_equal_length))

    data_blocks = iter(data_blocks)
    data = []
    data_length_all = [0] * comm.size  # all processes start from data=[]

    # repeat until all processes consume all data
    while True:
        # Read a block of data; we need to use `itertools.islice`
        # to support both of list-like objects and generators
        block = list(itertools.islice(data_blocks, block_size))

        # wrap the length by numpy array to communicate via MPI
        if block is None:
            block_length = numpy.array([0])
        else:
            block_length = numpy.array([len(block)])

        # How many elements does each process have?
        block_length_all = [x[0] for x in comm.allgather(block_length)]
        assert len(block_length_all) == comm.size
        assert len(data_length_all) == comm.size

        # If nobody has any more data to send. done.
        if all(n == 0 for n in block_length_all):
            # Process force_equal_length flag
            if force_equal_length:
                # Run _exchange_block one more time for force_equal_length
                data_length_all = _exchange_block(
                    comm, data, block, data_length_all, block_length_all,
                    True)
            break
        else:
            data_length_all = _exchange_block(
                comm, data, block, data_length_all, block_length_all,
                False)

    return data


def main():
    # test
    send_counts, new_length_all = _calc_alltoall_send_counts(
        [100, 100, 100, 100, 99, 99, 99, 99, 99],
        [20, 19, 20, 2, 13, 20, 20, 12, 3])
    print("send_counts = {}".format(send_counts))
    print("new_length_all = {}".format(new_length_all))

    send_counts, new_length_all = _calc_alltoall_send_counts(
        [100, 100, 100, 100, 99, 99, 99, 99, 99],
        [20, 19, 20, 2, 13, 20, 20, 12, 3],
        force_equal_length=True)
    print("send_counts = {}".format(send_counts))
    print("new_length_all = {}".format(new_length_all))

    send_counts, new_length_all = _calc_alltoall_send_counts(
        [100, 100, 100, 100, 99, 99, 99, 99, 99],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        force_equal_length=True)
    print("send_counts = {}".format(send_counts))
    print("new_length_all = {}".format(new_length_all))

    send_counts, new_length_all = _calc_alltoall_send_counts(
        [0, 0],
        [1, 10],
        force_equal_length=False)
    print("send_counts = {}".format(send_counts))
    print("new_length_all = {}".format(new_length_all))


if __name__ == '__main__':
    main()