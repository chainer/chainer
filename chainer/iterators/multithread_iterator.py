from __future__ import division
from multiprocessing import pool

import numpy
import six

from chainer.dataset import iterator
from chainer.iterators import random_state


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
        self.current_position = 0
        self.epoch = 0

        # Create a RandomState for each batch index.
        self._random_states = random_state.create_random_states(batch_size)
        # To support serialization and reset, we need to backup
        # the states as tuples until `__next__` is called.
        self._random_state_tuples = [state.get_state()
                                     for state in self._random_states]
        # Whether the random states are accessed during prefetch.
        self._random_dirties = [False] * batch_size

        self.n_threads = n_threads
        self._pool = None

        self.reset()

    def reset(self):
        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False
        if self._shuffle:
            self._order = numpy.random.permutation(len(self.dataset))
        else:
            self._order = None

        # reset internal state
        self._next = None
        self._previous_epoch_detail = None

    def __del__(self):
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
            # this is required when reset or serialized
            random_states = self._random_states
            random_state_tuples = self._random_state_tuples
            random_dirties = self._random_dirties
            for batch_idx in six.moves.range(self.batch_size):
                random_states[batch_idx].set_state(
                    random_state_tuples[batch_idx])
                random_dirties[batch_idx] = False

            # load for the first iteration
            self._invoke_prefetch()

        batch = self._get()

        random_states = self._random_states
        random_state_tuples = self._random_state_tuples
        random_dirties = self._random_dirties
        for batch_idx in six.moves.range(self.batch_size):
            if random_dirties[batch_idx]:
                random_state_tuples[batch_idx] = \
                    random_states[batch_idx].get_state()
                random_dirties[batch_idx] = False

        self._invoke_prefetch()  # prefetch for the next iteration
        return batch

    next = __next__

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / len(self.dataset)

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

        random_state_tuples = self._random_state_tuples
        try:
            for batch_idx in six.moves.range(self.batch_size):
                magic, key, pos, has_gauss, cached_gaussian = \
                    random_state_tuples[batch_idx]
                key_store = numpy.frombuffer(key, dtype=numpy.uint32)
                key_store[:] = serializer(
                    'batch_random/{}/key'.format(batch_idx), key_store)
                pos = serializer(
                    'batch_random/{}/pos'.format(batch_idx), pos)
                has_gauss = serializer(
                    'batch_random/{}/has_gauss'.format(batch_idx), has_gauss)
                cached_gaussian = serializer(
                    'batch_random/{}/cached_gaussian'.format(batch_idx),
                    cached_gaussian)
                random_state_tuples[batch_idx] = \
                    magic, key, pos, has_gauss, cached_gaussian
        except KeyError:
            pass

    @staticmethod
    def _read(args):
        dataset, dataset_idx, batch_idx, random_states, random_dirties = args

        def get_state():
            random_dirties[batch_idx] = True
            return random_states[batch_idx]

        with random_state.set_random_state(get_state):
            return dataset[dataset_idx]

    def _invoke_prefetch(self):
        assert self._next is None
        if not self._repeat and self.epoch > 0:
            return
        if self._pool is None:
            self._pool = pool.ThreadPool(self.n_threads)
        n = len(self.dataset)
        i = self.current_position

        order = self._order
        args = []
        dataset = self.dataset
        epoch = self.epoch
        is_new_epoch = False
        random_states = self._random_states
        random_dirties = self._random_dirties
        for batch_idx, _ in enumerate(six.moves.range(self.batch_size)):
            dataset_idx = i if order is None else order[i]
            args.append((dataset, dataset_idx, batch_idx,
                         random_states, random_dirties))
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
                    order = order.copy()
                    numpy.random.shuffle(order)

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
