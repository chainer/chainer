import warnings

import chainer.datasets
import numpy


class DataSizeError(RuntimeError):
    pass


def scatter_dataset(dataset, comm, root=0, shuffle=False,
                    seed=None, max_buf_len=256 * 1024 * 1024,
                    *, force_equal_length=True):
    """Scatter the given dataset to the workers in the communicator.

    The dataset of worker ``root``
    (i.e., the worker whose ``comm.rank`` is ``root``) is
    scattered to all workers. The given dataset of other workers are ignored.
    The dataset is split to sub datasets of almost equal sizes and scattered
    to workers. To create a sub dataset, ``chainer.datasets.SubDataset`` is
    used.

    Note::
        Make sure ``force_equal_length`` flag is *not* off for
        multinode evaluator or multinode updaters, which assume that
        the iterator has the same lengths among processes to work
        correctly.

    Args:
        dataset: A dataset (e.g., ``list``, ``numpy.ndarray``,
            ``chainer.datasets.TupleDataset``, ...).
        comm: ChainerMN communicator or MPI4py communicator.
        shuffle (bool): If ``True``, the order of examples is shuffled
            before being scattered.
        root (int): The root process of the scatter operation.
        seed (int): Seed the generator used for the permutation of indexes.
            If an integer being convertible to 32 bit unsigned integers is
            specified, it is guaranteed that each sample
            in the given dataset always belongs to a specific subset.
            If ``None``, the permutation is changed randomly.
        max_buf_len (int): Max buffer size to be used at broadcasting
            binaries. Must not be larger than 2147483647.
        force_equal_length (bool):
            Force the scattered fragments of the dataset have equal
            length. If ``True``, number of scattered examples is
            guaranteed to be equal among processes and scattered
            datasets may have duplication among processes. Otherwise,
            number of scattered examples may not be equal among
            processes, but scattered examples are guaranteed to have
            no duplication among processes, intended for strict
            evaluation of test dataset to avoid duplicated examples.

    Returns:
        Scattered dataset.

    """

    assert 0 <= root and root < comm.size

    order = None
    if shuffle and dataset is not None:
        n_total_samples = len(dataset)
        order = numpy.random.RandomState(seed).permutation(
            n_total_samples)

    data = (dataset, order) if comm.rank == root else None
    data = comm.bcast_obj(data, max_buf_len=max_buf_len, root=root)

    assert data is not None
    (dataset, order) = data

    (b, e) = scatter_index(
        len(dataset), comm, root,
        force_equal_length=force_equal_length)
    return chainer.datasets.SubDataset(dataset, b, e, order)


def scatter_index(n_total_samples, comm, root=0, *, force_equal_length=True):
    '''Scatters only index to avoid heavy dataset broadcast

    This is core functionality of ``scatter_dataset``, which is
    almost equal to following code snippet::

        (b, e) = scatter_index(len(dataset), comm)
        order = None
        if shuffle:
            order = numpy.random.RandomState(seed).permutation(
                n_total_samples)
            order = comm.bcast_obj(order)
        dataset = SubDataset(dataset, b, e, order)

    Note::
        Make sure ``force_equal_length`` flag is *not* off for
        multinode evaluator or multinode updaters, which assume that
        the iterator has the same lengths among processes to work
        correctly.

    Args:
        n_total_samples (int): number of total samples to scatter
        comm: ChainerMN communicator object
        root (int): root rank to coordinate the operation
        force_equal_length (bool):
            Force the scattered fragments of the index have equal
            length. If ``True``, number of scattered indices is
            guaranteed to be equal among processes and scattered
            datasets may have duplication among processes. Otherwise,
            number of scattered indices may not be equal among
            processes, but scattered indices are guaranteed to have
            no duplication among processes, intended for strict
            evaluation of test dataset to avoid duplicated examples.

    Returns:
        Tuple of two integers, that stands for beginning and ending
        offsets of the assigned sub part of samples. The ending offset
        is not border inclusive.

    '''
    if comm.rank == root:
        for (i, b, e) in _scatter_index(n_total_samples, comm.size,
                                        force_equal_length):
            if i == root:
                mine = (b, e)
            else:
                comm.send_obj((b, e), dest=i)
        return mine
    else:
        return comm.recv_obj(source=root)


def _scatter_index(n_total_samples, size, force_equal_length):
    assert size > 0
    assert n_total_samples >= 0
    if force_equal_length:
        n_sub_samples = (n_total_samples + size - 1) // size
        for i in range(size):
            b = n_total_samples * i // size
            e = b + n_sub_samples
            yield (i, b, e)
        return
    else:
        b = 0
        stride = (n_total_samples // size) + 1
        threshold = n_total_samples % size
        for i in range(threshold):
            e = b + stride
            yield (i, b, e)
            b += stride
        stride = n_total_samples // size
        for i in range(threshold, size):
            e = b + stride
            yield (i, b, e)
            b += stride
        return


def get_n_iterations_for_one_epoch(dataset, local_batch_size, comm):
    """Get the number of iterations for one epoch.

    .. note::

        This API is deprecated. Please use standard epoch triggers.

    Args:
        dataset: Sub dataset of each worker.
        local_batch_size (int): Batch size of each worker.
        comm: ChainerMN communicator or MPI4py communicator.

    Returns:
        int: the number of iterations for one epoch.
    """

    warnings.warn(
        'get_n_iterations_for_one_epoch is deprecated. Please use '
        'standard epoch triggers.', DeprecationWarning)

    n_iterations = None
    if comm.rank == 0:
        n_iterations = (len(dataset) + local_batch_size -
                        1) // local_batch_size
    return comm.bcast_obj(n_iterations)


def get_epoch_trigger(n_epochs, dataset, local_batch_size, comm):
    """Get the trigger that behaves like an epoch trigger.

    .. note::

        This API is deprecated. Please use standard epoch triggers.

    Args:
        n_epochs (int): The number of epochs.
        dataset: Sub dataset of each worker.
        local_batch_size (int): Batch size of each worker.
        comm: ChainerMN communicator or MPI4py communicator.

    Returns:
        The trigger that behaves like the epoch trigger.
    """

    warnings.warn(
        'get_epoch_trigger is deprecated. Please use standard epoch triggers.',
        DeprecationWarning)

    n_iterations = n_epochs * get_n_iterations_for_one_epoch(
        dataset, local_batch_size, comm)
    return n_iterations, 'iteration'
