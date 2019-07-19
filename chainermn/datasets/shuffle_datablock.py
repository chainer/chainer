import itertools
import numpy as np


def _count_table(length_all):
    """Returns a 2d numpy array of size NP x NP where NP is the comm_size,
    of which element [sender, receiver] is
    the send/recv count for send()/recv() call.

    i.e.
    sender rank i sends N elements to receiver j where N = table[i, j]
    """

    comm_size = len(length_all)
    table = np.zeros([comm_size, comm_size], dtype=int)

    # Distribute `length_all` elements over all processes
    # To achieve a good balance, the reminder is scattered over
    # processes of [rank, rank+1, ..., 0, 1]
    for send_rank in range(comm_size):
        sender_length = length_all[send_rank]
        q = sender_length // comm_size
        rem = sender_length % comm_size
        for recv_rank in range(comm_size):
            if (recv_rank - send_rank + comm_size) % comm_size < rem:
                table[send_rank, recv_rank] = q + 1
            else:
                table[send_rank, recv_rank] = q
        assert sum(table[send_rank, :]) == sender_length
    return table


def _send_recv_pairs(size):
    pairs = [(x, y) for x in range(size) for y in range(size)]
    # Sort the list, so send-recv communications from lower rank to higher rank
    # happen earlier to avoid deadlock
    # ex.)  [(0, 0), (1, 1), (0, 1), (1, 0)]
    #     --> [(0, 0), (0, 1), (1, 0), (1, 1)]
    pairs.sort()
    return pairs


def _exclusive_scan(array, dtype=None):
    """C++'s std::exclusive_scan"""
    if dtype is None:
        dtype = array.dtype
    z = np.zeros(len(array), dtype=dtype)
    z[1:] = np.cumsum(array, dtype=dtype)[:-1]

    return z


def shuffle_data_chunks(comm, data_chunks, force_equal_length=True, chunk_size=10000):
    """Exchange chunks of data between all processes,
    where the data is huge or the total length is unknown.

    :param comm: ChainerMN communicator
    :param data_chunks: list or generator to read chunks.
    :param force_equal_length: Whether data length of each process is adjusted
                               by copying some elements, so that all processes
                               have exactly the same number of training data.
                               This is required in training for correct
                               iteration/epoch counting.
                               In evaluation, however, it is not required and
                               the option can be False if you don't want
                               duplicated elements.
    :param chunk_size: Number of elements read from `data_chunks` at a time
    :return: List of shuffled data
    """

    if chunk_size <= 1:
        raise ValueError()

    if force_equal_length not in [None, True, False]:
        raise ValueError('Wrong value for `force_equal_length`:'
                         ' {}'.format(force_equal_length))

    data_chunks = iter(data_chunks)
    data = []

    if comm.rank == 0:
        print("\n\n======================", flush=True)
        print("chunk_size = {}".format(chunk_size))
        print("force_equal_length = {}".format(force_equal_length))

    # repeat until all processes consume all data
    while True:
        comm.mpi_comm.barrier()
        if comm.rank == 0:
            print("---------------------", flush=True)
        comm.mpi_comm.barrier()

        # Read a chunk of data
        chunk = list(itertools.islice(data_chunks, chunk_size))

        if chunk is None:
            length = np.array([0])
        else:
            length = np.array([len(chunk)])

        # print("Rank {}: length = {}".format(comm.rank, length))

        # How many elements does each process have?
        chunk_length_all = [x[0] for x in comm.allgather(length)]
        assert len(chunk_length_all) == comm.size
        if comm.rank == 0:
            print("chunk_length_all = {}".format(chunk_length_all), flush=True)

        # Nobody has any more data to send. done.
        if all(n == 0 for n in chunk_length_all):
            break

        count_table = _count_table(chunk_length_all)

        # Perform parallel communication (which is very similar to alltoallv)
        # using send() and recv()

        # First, generate all (sender, receiver) pairs
        size = comm.size
        pairs = _send_recv_pairs(comm.size)

        # offset of sendcounts
        offset = _exclusive_scan(count_table[comm.rank, :])
        # print("Rank {}: pairs = {}".format(comm.rank, pairs))

        # Perform send() and recv() in the order of `pairs`
        # i.e.
        #   pairs (0 --> 1) and (2 --> 3) are independent, so their
        #   communications happen simultaneously
        for send_rank, recv_rank in pairs:
            count = count_table[send_rank, recv_rank]

            if count == 0:
                continue

            if send_rank == recv_rank:
                if send_rank == comm.rank:
                    # self copy
                    beg = offset[recv_rank]
                    end = beg + count
                    data += chunk[beg:end]
            elif send_rank == comm.rank:
                assert recv_rank != comm.rank
                # I am the sender
                beg = offset[recv_rank]
                end = beg + count
                comm.send_obj(chunk[beg:end], dest=recv_rank, tag=0)
                # print("Rank {}: {} --> {}, data={}".format(comm.rank, send_rank, recv_rank, chunk[beg:end]), flush=True)
            elif recv_rank == comm.rank:
                # I am the receiver
                assert send_rank != comm.rank
                recv_data = comm.recv_obj(source=send_rank, tag=0)
                data += recv_data

        comm.mpi_comm.barrier()
        print("Rank {} len(data) = {}".format(comm.rank, len(data)), flush=True)
        comm.mpi_comm.barrier()

    # print("Rank {}: data={}".format(comm.rank, data))
    if comm.rank == 0:
        print("Rank {}: finished main loop".format(comm.rank), flush=True)
    comm.mpi_comm.barrier()
    print("Rank {}: len(data) = {}".format(comm.rank, len(data)))

    if force_equal_length:
        # After all communications, get the length of data of all ranks
        _adjust_data_length(comm, data)

    comm.mpi_comm.barrier()
    if comm.rank == 0:
        print("Rank {}: finished force_equal_length".format(comm.rank), flush=True)
    return data


def _adjust_data_length(comm, data):
    length_all = [x[0] for x in comm.allgather(np.array([len(data)]))]
    length_final = max(length_all)
    print("length_final = {}".format(length_final))

    # The process which has maximum number of elements sends its data
    # to other processes

    root_rank = next(r for r in range(comm.size)
                     if length_all[r] == length_final)
    print("root_rank = {}".format(root_rank))

    if comm.rank == root_rank:
        offset = 0
        for dest_rank in range(comm.size):
            if comm.rank == dest_rank:
                continue

            send_cnt = length_final - length_all[dest_rank]
            if send_cnt > 0:
                print(
                    "Rank {}: Sending {} elements to rank {}".format(comm.rank,
                                                                     send_cnt,
                                                                     dest_rank))
                comm.send_obj(data[offset:offset + send_cnt], dest_rank, tag=0)
                offset += send_cnt
    else:
        if len(data) < length_final:
            print("Rank {}: recv from rank {}".format(comm.rank, root_rank))
            data += comm.recv_obj(source=root_rank, tag=0)
        assert len(data) == length_final


def test_count_table():
    from numpy.testing import assert_array_equal

    ## --
    length_all = [0, 0, 100]
    answer = np.array([[ 0,  0,  0],
                       [ 0,  0,  0],
                       [33, 33, 34]])
    assert_array_equal(_count_table(length_all), answer)

    ## --
    length_all = [0, 0, 101]
    answer = np.array([[0, 0, 0],
                       [0, 0, 0],
                       [34, 33, 34]])
    assert_array_equal(_count_table(length_all), answer)

    ## --
    length_all = [100]
    answer = np.array([[100]])
    assert_array_equal(_count_table(length_all), answer)

    ## --
    length_all = [10, 20, 30]
    answer = np.array([[4, 3, 3],
                       [6, 7, 7],
                       [10, 10, 10]])
    assert_array_equal(_count_table(length_all), answer)


import pytest


@pytest.mark.parametrize('chunk_size,force_equal_length',
                         list(itertools.product([1000, 100000],
                                                [True, False])))
def test_shuffle_datablocks(chunk_size, force_equal_length):
    import chainermn
    comm = chainermn.create_communicator('pure_nccl')

    # Rank i generates data = range(10**i)
    num_elem = 10 ** (comm.rank + 1)
    data = range(num_elem, num_elem *2 + 3)

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
    from numpy.testing import assert_array_equal

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


if __name__ == '__main__':
    import numpy as np
    import chainermn
    import time
    import random
    comm = chainermn.communicators.create_communicator()

    r = range(comm.rank * 10, comm.rank * 10 + (comm.rank + 5) * 100 + random.randint(10, 500))

    for i in range(comm.size):
        if i == comm.rank:
            if i == 0:
                print("------------------------------")

            print(len(r))
        comm.mpi_comm.barrier()

    chunks = [np.array([i]) for i in r]
    data = shuffle_data_chunks(comm, chunks, force_equal_length=True)
    data = [x[0] for x in data]

    time.sleep(0.5)

    for i in range(comm.size):
        if i == comm.rank:
            if i == 0:
                print("------------------------------")

            # print("{}, {}".format(data, len(data)), flush=True)
            print("{}".format(len(data)), flush=True)
        comm.mpi_comm.barrier()

    time.sleep(0.5)

    length_all = [x[0] for x in comm.allgather(np.array([len(data)]))]
    if comm.rank == 0:
        print()
        print(length_all)
        print("max = {}".format(max(length_all)))
        print("min = {}".format(min(length_all)))
        print("dif = {}".format(max(length_all) - min(length_all)))

