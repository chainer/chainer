from __future__ import division
from multiprocessing import pool
import threading

import numpy
import six

from chainer.dataset import iterator


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
            order of indexes.
        n_threads (int): Number of worker threads.

    """

    def __init__(self, dataset, batch_size, repeat=True, shuffle=True,
                 n_threads=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self._repeat = repeat
        self._shuffle = shuffle
        self._prefetch_order = None  # used at the end of each epoch

        self.n_threads = n_threads

        self._finalized = None

        self.reset()

    def reset(self):
        if getattr(self, 'current_position', 0) != 0:
            raise NotImplementedError(
                'Reset of MultithreadIterator in the middle of a epoch is '
                'currently not supported.')
        if getattr(self, 'epoch', 0) != 0 and self._repeat:
            raise NotImplementedError(
                'Reset of repeating MultithreadIterator is currently not '
                'supported.')
        if (getattr(self, '_finalized', None) is not None and
                self._finalized.is_set()):
            raise NotImplementedError(
                'Reset of finalized MultithreadIterator is currently not '
                'supported.')

        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False

        self._previous_epoch_detail = None

        self._pushed_position = None  # initialized at the first iteration

        if self._shuffle:
            self._order = numpy.random.permutation(len(self.dataset))
        else:
            self._order = None

        if self._finalized is not None:
            self._next = None
            self._invoke_prefetch()

    def __del__(self):
        self.finalize()

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        self._previous_epoch_detail = self.epoch_detail

        self.is_new_epoch = False
        if self._finalized is None:
            self._init()  # start workers
            # load for the first iteration
            self._invoke_prefetch()

        batch = self._get()
        self._invoke_prefetch()  # prefetch for the next iteration
        return batch

    next = __next__

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / len(self.dataset)

    @property
    def previous_epoch_detail(self):
        return self._previous_epoch_detail

    def finalize(self):
        if self._finalized is None or self._finalized.is_set():
            return

        self._finalized.set()
        self._next = None
        self._pool.close()
        self._pool.join()

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        try:
            serializer('order', self._order)
        except KeyError:
            serializer('_order', self._order)
        try:
            self._previous_epoch_detail = serializer(
                'previous_epoch_detail', self._previous_epoch_detail)
        except KeyError:
            # guess previous_epoch_detail for older version
            self._previous_epoch_detail = (
                self.epoch +
                (self.current_position - self.batch_size) / len(self.dataset))
            if self.epoch_detail > 0:
                if self._previous_epoch_detail is None:
                    self._previous_epoch_detail = 0.
            else:
                self._previous_epoch_detail = None

    def _init(self):
        finalized = threading.Event()
        self._index_queue = six.moves.queue.Queue()
        self._data_queue = six.moves.queue.Queue()
        self._cnt = 0

        self._workers = []

        self._finalized = finalized

        self._pool = pool.ThreadPool(self.n_threads)
        self._next = None

    def _invoke_prefetch(self):
        assert self._next is None
        n = len(self.dataset)
        i = self._pushed_position
        if i is None:  # first iteration
            i = self.current_position

        order = self._order
        args = []
        for _ in six.moves.range(self.batch_size):
            if i >= n:
                if not self._repeat:
                    break
                i = 0
                if order is not None:
                    # We cannot shuffle the order directly here, since the
                    # iterator may be serialized before the prefetched data are
                    # consumed by the user, in which case an inconsistency
                    # appears.
                    order = order.copy()
                    numpy.random.shuffle(order)
            index = i if order is None else order[i]
            args.append(index)
            i += 1

        dataset = self.dataset

        def _read(index):
            return dataset[index]

        self._next = self._pool.map_async(_read, args)
        self._prefetch_order = order  # Temporarily store the shuffled order.
        self._pushed_position = i

    def _get(self):
        n = len(self.dataset)
        i = self.current_position

        batch = []
        next = self._next
        while not next.ready():
            next.wait(0.5)  # To avoid interruption bug in Python2

        for data in next.get():
            batch.append(data)
            i += 1
            if i >= n:
                self.epoch += 1
                self.is_new_epoch = True
                i = 0
                if not self._repeat:
                    break
        self._next = None

        self.current_position = i
        # Eventually overwrite the (possibly shuffled) order.
        self._order = self._prefetch_order
        return batch
