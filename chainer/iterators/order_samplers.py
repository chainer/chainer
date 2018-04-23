import numpy


class ShuffleOrderSampler(object):

    """Sampler that generates random orders.

    This is expected to be used together with Chainer's iterators.
    An order sampler is called by an iterator every epoch.

    The two lines below create basically the same objects.

    >>> chainer.iterators.MultiProcessIterator(dataset, 1, shuffle=True)
    >>> chainer.iterators.MultiProcessIterator(
    ...     dataset, 1, order_sampler=ShuffleOrderSampler())

    """

    def __init__(self, seed=None):
        # Use a distinct RandomState in the thread
        # for deterministic random number generation.
        # To support 32-bit platform and numpy < 1.11,
        # the seed is taken in a verbose manner.
        if seed is None:
            seed = numpy.random.randint(2**31)
        self._random = numpy.random.RandomState(seed)

    def __call__(self, current_order, current_position):
        """Sample the next order.

        Args:
            current_order (numpy.ndarray): 1-D array of indices.
                The length should be the same as the dataset to sample
                data from.
            current_position (int): The current position of an iterator.

        Returns:
            numpy.ndarray: 1-D array of indices. This is the order in which
                examples are sampled from a dataset in the next epoch.

        """
        return self._random.permutation(len(current_order))
