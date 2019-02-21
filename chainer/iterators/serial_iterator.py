from __future__ import division

import numpy

from chainer.dataset import iterator
from chainer.iterators import _statemachine
from chainer.iterators.order_samplers import ShuffleOrderSampler


class SerialIterator(iterator.Iterator):

    """Dataset iterator that serially reads the examples.

    This is a simple implementation of :class:`~chainer.dataset.Iterator`
    that just visits each example in either the order of indexes or a shuffled
    order.

    To avoid unintentional performance degradation, the ``shuffle`` option is
    set to ``True`` by default. For validation, it is better to set it to
    ``False`` when the underlying dataset supports fast slicing. If the
    order of examples has an important meaning and the updater depends on the
    original order, this option should be set to ``False``.

    This iterator saves ``-1`` instead of ``None`` in snapshots since some
    serializers do not support ``None``.

    Args:
        dataset: Dataset to iterate.
        batch_size (int): Number of examples within each batch.
        repeat (bool): If ``True``, it infinitely loops over the dataset.
            Otherwise, it stops iteration at the end of the first epoch.
        shuffle (bool): If ``True``, the order of examples is shuffled at the
            beginning of each epoch. Otherwise, examples are extracted in the
            order of indexes. If ``None`` and no ``order_sampler`` is given,
            the behavior is the same as the case with ``shuffle=True``.
        order_sampler (callable): A callable that generates the order
            of the indices to sample in the next epoch when a epoch finishes.
            This function should take two arguments: the current order
            and the current position of the iterator.
            This should return the next order. The size of the order
            should remain constant.
            This option cannot be used when ``shuffle`` is not ``None``.

    """

    def __init__(self, dataset, batch_size,
                 repeat=True, shuffle=None, order_sampler=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self._repeat = repeat
        self._shuffle = shuffle

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

        self.reset()

    def __next__(self):
        self._previous_epoch_detail = self.epoch_detail
        self._state, indices = _statemachine.iterator_statemachine(
            self._state, self.batch_size, self.repeat, self.order_sampler,
            len(self.dataset))
        if indices is None:
            raise StopIteration

        batch = [self.dataset[index] for index in indices]
        return batch

    next = __next__

    @property
    def current_position(self):
        return self._state.current_position

    @property
    def epoch(self):
        return self._state.epoch

    @property
    def is_new_epoch(self):
        return self._state.is_new_epoch

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / self._epoch_size

    @property
    def previous_epoch_detail(self):
        # use -1 instead of None internally.
        if self._previous_epoch_detail < 0:
            return None
        return self._previous_epoch_detail

    def serialize(self, serializer):
        current_position = serializer('current_position',
                                      self.current_position)
        epoch = serializer('epoch', self.epoch)
        is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        order = self._state.order
        if order is not None:
            try:
                serializer('order', order)
            except KeyError:
                serializer('_order', order)
        self._state = _statemachine.IteratorState(
            current_position, epoch, is_new_epoch, order)
        try:
            self._previous_epoch_detail = serializer(
                'previous_epoch_detail', self._previous_epoch_detail)
        except KeyError:
            # guess previous_epoch_detail for older version
            self._previous_epoch_detail = self.epoch + \
                (self.current_position - self.batch_size) / self._epoch_size
            if self.epoch_detail > 0:
                self._previous_epoch_detail = max(
                    self._previous_epoch_detail, 0.)
            else:
                self._previous_epoch_detail = -1.

    def reset(self):
        if self.order_sampler:
            order = self.order_sampler(
                numpy.arange(len(self.dataset)), 0)
        else:
            order = None
        self._state = _statemachine.IteratorState(0, 0, False, order)
        self._previous_epoch_detail = -1.

    @property
    def _epoch_size(self):
        order = self._state.order
        if order is None:
            epoch_size = len(self.dataset)
        else:
            epoch_size = len(order)
        return epoch_size

    @property
    def repeat(self):
        return self._repeat
