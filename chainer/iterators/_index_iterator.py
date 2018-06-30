import numpy

from chainer.dataset import iterator


class IndexIterator(iterator.Iterator):
    """Index iterator

    `IndexIterator` is used internally in `OrderSampler`
    (e.g., `BalancedOrderSampler`), as each label's index iterator

        Args:
            index_list (list or numpy.ndarray): list of int which represents
                indices.
            shuffle (bool): shuffle flag. If True, indices specified by
                `index_list` will be randomly shuffled.
            num (int): number of indices to be extracted when `___next___` is
                called.

    """

    def __init__(self, index_list, shuffle=True, num=0):
        self.index_list = numpy.asarray(index_list)
        if self.index_list.ndim != 1:
            raise ValueError("[ERROR] index_list must be 1-dim list or array")
        self.index_length = len(index_list)
        self.current_index_list = None
        self.current_pos = 0
        self.shuffle = shuffle
        self.num = num

        self.update_current_index_list()

    def update_current_index_list(self):
        if self.shuffle:
            self.current_index_list = numpy.random.permutation(self.index_list)
        else:
            self.current_index_list = self.index_list

    def __next__(self):
        return self.get_next_indices(self.num)

    def get_next_indices(self, num):
        """get next indices

        Args:
            num (int): number for indices to extract.

        Returns (numpy.ndarray): 1d array of indices

        .. admonition:: Example

           >>> ii = IndexIterator([1, 3, 5, 10], shuffle=True)
           >>> print(ii.get_next_indices(5))
           [ 5  1 10  3 10]
           >>> print(ii.get_next_indices(5))
           [ 3  1  5 10  1]

        """

        indices = []
        if self.current_pos + num < self.index_length:
            indices.append(self.current_index_list[
                           self.current_pos: self.current_pos + num])
            self.current_pos += num
        else:
            indices.append(self.current_index_list[self.current_pos:])
            num -= (self.index_length - self.current_pos)
            # When `num` is twice bigger than `self.index_length`, `index_list`
            # is repeated `q` times to get desired length of `indices`.
            q, r = divmod(num, self.index_length)
            if self.shuffle:
                for _ in range(q):
                    indices.append(numpy.random.permutation(self.index_list))
            else:
                indices.append(numpy.tile(self.index_list, q))
            self.update_current_index_list()
            indices.append(self.current_index_list[:r])
            self.current_pos = r

        return numpy.concatenate(indices).ravel()

    def serialize(self, serializer):
        self.current_index_list = serializer('current_index_list',
                                             self.current_index_list)
        self.current_pos = serializer('current_pos', self.current_pos)
