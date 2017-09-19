import numpy


def shuffle_order_sampler(current_order, current_position):
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
    return numpy.random.permutation(len(current_order))


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
