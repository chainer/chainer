import numpy


class ShuffleOrderSampler(object):

    def __init__(self, seed=None):
        # Use a distinct RandomState in the thread
        # for deterministic random number generation.
        # To support 32-bit platform and numpy < 1.11,
        # the seed is taken in a verbose manner.
        if seed is None:
            seed = numpy.random.randint(2**31)
        self._random = numpy.random.RandomState(seed)

    def __call__(self, current_order, current_position):
        """The order sampler of an iterator when shuffle option is True.

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


def no_shuffle_order_sampler(current_order, current_position):
    """The default order sampler of an iterator when shuffle option is False.

    Args:
        current_order (numpy.ndarray): 1-D array of indices.
            The length should be the same as the dataset to sample
            data from.
        current_position (int): The current position of an iterator.

    Returns:
        None:

    """
    return None
