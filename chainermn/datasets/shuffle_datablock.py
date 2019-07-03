import itertools
import numpy as np


def _count_table(length_all):
    """Returns a 2d numpy array, of which [sender, receiver] is
    the sendcount and recvcount for send()/recv() call"""

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
    # happen earlier
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


def shuffle_data_chunks(comm, data_chunks, force_equal_length='copy', chunk_size=10000):
    if force_equal_length not in ['copy', 'drop', None, False]:
        raise ValueError('Wrong value for `force_equal_length`:'
                         ' {}'.format(force_equal_length))

    data_chunks = iter(data_chunks)
    data = []

    # repeat until all processes consume all data
    while True:
        comm.mpi_comm.barrier()
        if comm.rank == 0:
            print("---------------------", flush=True)
        comm.mpi_comm.barrier()

        chunk = list(itertools.islice(data_chunks, chunk_size))

        if chunk is None:
            length = np.array([0])
        else:
            length = np.array([len(chunk)])

        print("Rank {}: length = {}".format(comm.rank, length))
        length_all = [x[0] for x in comm.allgather(length)]
        assert len(length_all) == comm.size
        print("Rank {}: length_all = {}".format(comm.rank, length_all))

        if all(n == 0 for n in length_all):
            # Nobody has any more data to send. done.
            break

        count_table = _count_table(length_all)

        # Perform parallel communication (which is very similar to alltoallv)
        # using send() and recv()

        # First, generate all (sender, receiver) pairs
        size = comm.size
        pairs = _send_recv_pairs(comm.size)

        # offset of sendcounts
        offset = _exclusive_scan(count_table[comm.rank, :])
        # print("Rank {}: pairs = {}".format(comm.rank, pairs))

        for send_rank, recv_rank in pairs:
            count = count_table[send_rank, recv_rank]

            if count == 0:
                continue

            if send_rank == recv_rank:
                if send_rank == comm.rank:
                    beg = offset[recv_rank]
                    end = beg + count
                    data += chunk[beg:end]
            elif send_rank == comm.rank:
                beg = offset[recv_rank]
                end = beg + count
                comm.send(chunk[beg:end], dest=recv_rank, tag=0)
                # print("Rank {}: {} --> {}, data={}".format(comm.rank, send_rank, recv_rank, chunk[beg:end]), flush=True)
            elif recv_rank == comm.rank:
                recv_data = comm.recv(source=send_rank, tag=0)
                data += recv_data

            comm.mpi_comm.barrier()

    # print("Rank {}: data={}".format(comm.rank, data))
    length_all = [x[0] for x in comm.allgather(np.array([len(data)]))]

    if force_equal_length == 'drop':
        shortest = min(length_all)
        data = data[:shortest]
    elif force_equal_length == 'copy':
        raise NotImplemented()

    return data


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


def test_send_recv_pairs():
    from numpy.testing import assert_array_equal

    # 3x3 table
    answer = [(0, 0), (2, 1), (1, 2),
              (0, 1), (2, 2), (1, 0),
              (0, 2), (2, 0), (1, 1)]
    assert_array_equal(_send_recv_pairs(3), answer)

    # 4x4 table
    answer = [(0, 0), (3, 1), (2, 2), (1, 3),
              (0, 1), (3, 2), (2, 3), (1, 0),
              (0, 2), (3, 3), (2, 0), (1, 1),
              (0, 3), (3, 0), (2, 1), (1, 2)]
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
    data = shuffle_data_chunks(comm, chunks, force_equal_length='drop')
    data = [x[0] for x in data]

    time.sleep(0.5)

    for i in range(comm.size):
        if i == comm.rank:
            if i == 0:
                print("------------------------------")

            # print("{}, {}".format(data, len(data)), flush=True)
            print("{}".format(len(data)), flush=True)
        comm.mpi_comm.barrier()

    max_len = comm.allreduce

    time.sleep(0.5)

    length_all = [x[0] for x in comm.allgather(np.array([len(data)]))]
    if comm.rank == 0:
        print()
        print(length_all)
        print("max = {}".format(max(length_all)))
        print("min = {}".format(min(length_all)))
        print("dif = {}".format(max(length_all) - min(length_all)))

