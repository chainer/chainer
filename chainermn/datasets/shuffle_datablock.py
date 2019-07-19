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


def shuffle_data_chunks(comm, data_chunks, force_equal_length=True,
                        chunk_size=10000):
    """Exchange chunks of data between all processes,
    where the data is huge or the total length is unknown.

    :param comm: ChainerMN communicator
    :param data_chunks: Sequence or generator to read chunks.
    :param force_equal_length: Whether data length of each process is adjusted
                               by copying some elements, so that all processes
                               have exactly the same length of data.
                               This is required in training for correct
                               iteration/epoch counting.
                               In evaluation, however, the option can be False
                               if you don't want duplicated elements.
    :param chunk_size: Number of elements read from `data_chunks` at a time
    :return: Shuffled data (in a list)
    """

    if chunk_size <= 1:
        raise ValueError()

    if force_equal_length not in [None, True, False]:
        raise ValueError('Wrong value for `force_equal_length`:'
                         ' {}'.format(force_equal_length))

    data_chunks = iter(data_chunks)
    data = []

    # repeat until all processes consume all data
    while True:
        # Read a chunk of data
        chunk = list(itertools.islice(data_chunks, chunk_size))

        if chunk is None:
            length = np.array([0])
        else:
            length = np.array([len(chunk)])

        # How many elements does each process have?
        chunk_length_all = [x[0] for x in comm.allgather(length)]
        assert len(chunk_length_all) == comm.size

        # Nobody has any more data to send. done.
        if all(n == 0 for n in chunk_length_all):
            break

        count_table = _count_table(chunk_length_all)

        # Perform parallel communication (which is very similar to alltoallv)
        # using send() and recv()

        # Generate all (sender, receiver) pairs
        size = comm.size
        pairs = _send_recv_pairs(comm.size)

        # offset of sendcounts
        offset = _exclusive_scan(count_table[comm.rank, :])

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
            elif recv_rank == comm.rank:
                # I am the receiver
                assert send_rank != comm.rank
                recv_data = comm.recv_obj(source=send_rank, tag=0)
                data += recv_data

    if force_equal_length:
        # After all communications, get the length of data of all ranks
        _adjust_data_length(comm, data)

    return data


def _adjust_data_length(comm, data):
    # The process which has maximum number of elements sends its data
    # to other processes

    length_all = [x[0] for x in comm.allgather(np.array([len(data)]))]
    length_final = max(length_all)

    root_rank = next(r for r in range(comm.size)
                     if length_all[r] == length_final)

    if comm.rank == root_rank:
        offset = 0
        for dest_rank in range(comm.size):
            if comm.rank == dest_rank:
                continue

            send_cnt = length_final - length_all[dest_rank]
            if send_cnt > 0:
                comm.send_obj(data[offset:offset + send_cnt], dest_rank, tag=0)
                offset += send_cnt
    else:
        if len(data) < length_final:
            data += comm.recv_obj(source=root_rank, tag=0)
        assert len(data) == length_final


import pytest
