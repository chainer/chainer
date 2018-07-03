import numpy

from chainer.iterators.order_samplers.order_sampler import OrderSampler


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
        random_state (numpy.random.RandomState or None): Pseudo-random number
            generator.

    """

    def __init__(self, random_state=None):
        if random_state is None:
            random_state = numpy.random.random.__self__
        self._random = random_state

    def __call__(self, current_order, current_position):
        return self._random.permutation(len(current_order))

    def serialize(self, serializer):
        # nothing need to be serialized
        pass
