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
