import chainer
import numpy

from chainer.iterators._index_iterator import IndexIterator  # NOQA
from chainer.iterators.order_samplers.order_sampler import OrderSampler  # NOQA


class BalancedOrderSampler(OrderSampler):

    """Sampler that generates orders with balancing class label.

    This is expected to be used together with Chainer's iterators.
    An order sampler is called by an iterator every epoch.

    The two initializations below create basically the same objects.

    >>> dataset = [(1, 2), (3, 4)]
    >>> it = chainer.iterators.MultiprocessIterator(
    ...     dataset, 1, order_sampler=chainer.iterators.BalancedOrderSampler())

    Args:
        labels (list or numpy.ndarray): 1d array which specifies label feature
            of `dataset`. Its size must be same as the length of `dataset`.
        random_state (numpy.random.RandomState or None): Pseudo-random number
            generator.
        shuffle (bool): If ``True``, the order of examples is shuffled at the
            beginning of each epoch.
            Otherwise, the order is permanently same as that of `dataset`.
        batch_balancing (bool):  If ``True``, examples are sampled in the way
            that each label examples are roughly evenly sampled in each
            minibatch. Otherwise, the iterator only guarantees that total
            numbers of examples are same among label features.
        ignore_labels (int or list or None): Labels to be ignored.
            If not ``None``, the example whose label is in `ignore_labels`
            are not sampled by this iterator.
    """

    def __init__(self, labels, random_state=None, shuffle=True,
                 batch_balancing=False, ignore_labels=None):
        if random_state is None:
            random_state = numpy.random.random.__self__
        self._random = random_state
        # --- Initialize index_iterators ---
        # assert len(dataset) == len(labels)
        labels = numpy.asarray(labels)
        # if len(dataset) != labels.size:
        #     raise ValueError('dataset length {} and labels size {} must be '
        #                      'same!'.format(len(dataset), labels.size))
        labels = numpy.ravel(labels)
        self.labels = labels
        if ignore_labels is None:
            ignore_labels = []
        elif isinstance(ignore_labels, int):
            ignore_labels = [ignore_labels, ]
        self.ignore_labels = list(ignore_labels)
        self._shuffle = shuffle
        self._batch_balancing = batch_balancing
        self.labels_iterator_dict = {}

        max_label_count = -1
        include_label_count = 0
        for label in numpy.unique(labels):
            label_index = numpy.argwhere(labels == label).ravel()
            label_count = len(label_index)
            ii = IndexIterator(label_index, shuffle=shuffle)
            self.labels_iterator_dict[label] = ii
            if label in self.ignore_labels:
                continue
            if max_label_count < label_count:
                max_label_count = label_count
            include_label_count += 1

        self.max_label_count = max_label_count
        self.N_augmented = max_label_count * include_label_count

    def __call__(self, current_order, current_position):
        indices_list = []
        for label, index_iterator in self.labels_iterator_dict.items():
            if label in self.ignore_labels:
                # Not include index of ignore_labels
                continue
            indices_list.append(index_iterator.get_next_indices(
                self.max_label_count))

        if self._batch_balancing:
            # `indices_list` contains same number of indices of each label.
            # we can `transpose` and `ravel` it to get each label's index in
            # sequence, which guarantees that label in each batch is balanced.
            indices = numpy.array(indices_list).transpose().ravel()
            return indices
        else:
            indices = numpy.array(indices_list).ravel()
            return self._random.permutation(indices)

    def serialize(self, serializer):
        for label, index_iterator in self.labels_iterator_dict.items():
            self.labels_iterator_dict[label].serialize(
                serializer['index_iterator_{}'.format(label)])
