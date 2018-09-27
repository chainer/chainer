from __future__ import division
from multiprocessing import pool

import numpy
import six

from chainer.dataset import iterator
from chainer.iterators.order_samplers import ShuffleOrderSampler


class MultithreadIterator(iterator.Iterator):

    """Dataset iterator that loads examples in parallel.

    This is an implementation of :class:`~chainer.dataset.Iterator` that loads
    examples with worker threads. It uses the standard :mod:`threading`
    module to parallelize the loading.

    Note that this iterator effectively prefetches the examples for the next
    batch asynchronously after the current batch is returned.

    This iterator saves ``-1`` instead of ``None`` in snapshots since some
    serializers do not support ``None``.

    Args:
        dataset (~chainer.dataset.Dataset): Dataset to iterate.
        batch_size (int): Number of examples within each batch.
        repeat (bool): If ``True``, it infinitely loops over the dataset.
            Otherwise, it stops iteration at the end of the first epoch.
        shuffle (bool): If ``True``, the order of examples is shuffled at the
            beginning of each epoch. Otherwise, examples are extracted in the
            order of indexes. If ``None`` and no ``order_sampler`` is given,
            the behavior is the same as the case with ``shuffle=True``.
        n_threads (int): Number of worker threads.
        order_sampler (callable): A callable that generates the order
            of the indices to sample in the next epoch when a epoch finishes.
            This function should take two arguements: the current order
            and the current position of the iterator.
            This should return the next order. The size of the order
            should remain constant.
            This option cannot be used when ``shuffle`` is not ``None``.

    """

    def __init__(self, dataset, batch_size, repeat=True, shuffle=None,
                 n_threads=1, order_sampler=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self._repeat = repeat
        self._shuffle = shuffle
        self._prefetch_order = None  # used at the end of each epoch
        self.current_position = 0
        self.epoch = 0

        if self._shuffle is not None:
            if order_sampler is not None:
                raise ValueError('`shuffle` is not `None` and a custom '
                                 '`order_sampler` is set. Please set '
                                 '`shuffle` to `None` to use the custom '
                                 'order sampler.')
            else:
                if self._shuffle:
                    order_sampler = ShuffleOrderSampler()
        else:
            if order_sampler is None:
                order_sampler = ShuffleOrderSampler()
        self.order_sampler = order_sampler

        self.n_threads = n_threads
        self._pool = None

        self.reset()

    def reset(self):
        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False
        if self.order_sampler:
            self._order = self.order_sampler(
                numpy.arange(len(self.dataset)), 0)
        else:
            self._order = None

        # reset internal state
        self._next = None
        self._previous_epoch_detail = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.finalize()

    def finalize(self):
        pool = self._pool

        self._next = None
        self._pool = None
        if pool is not None:
            pool.terminate()

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        self._previous_epoch_detail = self.epoch_detail

        if self._next is None:
            # load for the first iteration
            self._invoke_prefetch()

        batch = self._get()
        self._invoke_prefetch()  # prefetch for the next iteration
        return batch

    next = __next__

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / self._epoch_size

    @property
    def previous_epoch_detail(self):
        return self._previous_epoch_detail

    def serialize(self, serializer):
        self.current_position = serializer(
            'current_position', self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        self._order = serializer('_order', self._order)
        self._previous_epoch_detail = serializer(
            'previous_epoch_detail', self._previous_epoch_detail)
        self._next = None

    @staticmethod
    def _read(args):
        dataset, index = args
        return dataset[index]

    def _invoke_prefetch(self):
        assert self._next is None
        if not self._repeat and self.epoch > 0:
            return
        if self._pool is None:
            self._pool = pool.ThreadPool(self.n_threads)
        n = self._epoch_size
        i = self.current_position

        order = self._order
        args = []
        dataset = self.dataset
        epoch = self.epoch
        is_new_epoch = False
        for _ in six.moves.range(self.batch_size):
            index = i if order is None else order[i]
            args.append((dataset, index))
            i += 1
            if i >= n:
                epoch += 1
                is_new_epoch = True
                i = 0
                if not self._repeat:
                    break
                if order is not None:
                    # We cannot shuffle the order directly here, since the
                    # iterator may be serialized before the prefetched data are
                    # consumed by the user, in which case an inconsistency
                    # appears.
                    new_order = self.order_sampler(order, i)
                    if len(new_order) != len(order):
                        raise ValueError('The size of order does not match '
                                         'the size of the previous order.')
                    order = new_order

        self._next = self._pool.map_async(MultithreadIterator._read, args)
        self._next_state = (i, epoch, is_new_epoch, order)

    def _get(self):
        next = self._next
        while not next.ready():
            next.wait(0.5)  # To avoid interruption bug in Python2

        batch = [data for data in next.get()]
        self._next = None

        (self.current_position, self.epoch,
         self.is_new_epoch, self._order) = self._next_state
        return batch

    @property
    def _epoch_size(self):
        if self._order is None:
            epoch_size = len(self.dataset)
        else:
            epoch_size = len(self._order)
        return epoch_size

    @property
    def repeat(self):
        return self._repeat
