import numpy


class OrderSampler(object):

    """Base class of all order samplers.

    Every order sampler subclass has to provide a method
    :meth:`__call__`.
    This method is called by an iterator before a new epoch,
    and it should return a new index order for the next epoch.

    """

    def __call__(self, current_order, current_position):
        """Sample the next order.

        Args:
            current_order (numpy.ndarray): 1-D array of indices.
                The length should be the same as the dataset to sample
                data from.
            current_position (int): The current position of an iterator.

        Returns:
            numpy.ndarray:
            1-D array of indices. This is the order in which
            examples are sampled from a dataset in the next epoch.

        """
        raise NotImplementedError


class ShuffleOrderSampler(OrderSampler):

    """Sampler that generates random orders.

    This is expected to be used together with Chainer's iterators.
    An order sampler is called by an iterator every epoch.

    The two initializations below create basically the same objects.

    >>> dataset = [(1, 2), (3, 4)]
    >>> it = chainer.iterators.MultiprocessIterator(dataset, 1, shuffle=True)
    >>> it = chainer.iterators.MultiprocessIterator(
    ...     dataset, 1, order_sampler=chainer.iterators.ShuffleOrderSampler())

    Args:
        random_state (numpy.random.RandomState): Pseudo-random number
            generator.

    """

    def __init__(self, random_state=None):
        if random_state is None:
            random_state = numpy.random.random.__self__
        self._random = random_state

    def __call__(self, current_order, current_position):
        return self._random.permutation(len(current_order))
