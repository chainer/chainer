from chainer.dataset.tabular import tabular_dataset


class DelegateDataset(tabular_dataset.TabularDataset):

    """A helper class to implement a TabularDataset.

    This class wraps an instance of :class:`~chainer.dataset.TabularDataset`
    and provides methods of :class:`~chainer.dataset.TabularDataset`.
    This class is useful to create a custom dataset class by inheriting it.

    >>> import numpy as np
    >>>
    >>> from chainer.dataset import tabular
    >>>
    >>> class MyDataset(tabular.DelegateDataset):
    ...
    ...     def __init__(self):
    ...         super().__init__(tabular.from_data((
    ...             ('a', np.arange(10)),
    ...             ('b', self.get_b),
    ...             ('c', [3, 1, 4, 5, 9, 2, 6, 8, 7, 0]),
    ...             (('d', 'e'), self.get_de))))
    ...
    ...     def get_b(self, i):
    ...         return 'b[{}]'.format(i)
    ...
    ...     def get_de(self, i):
    ...         return {'d': 'd[{}]'.format(i), 'e': 'e[{}]'.format(i)}
    ...
    >>> dataset = MyDataset()
    >>> len(dataset)
    10
    >>> dataset.keys
    ('a', 'b', 'c', 'd', 'e')
    >>> dataset[0]
    (0, 'b[0]', 3, 'd[0]', 'e[0]')

    Args:
        dataset (chainer.dataset.TabularDataset): An underlying dataset.

    """

    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    @property
    def keys(self):
        return self._dataset.keys

    @keys.setter
    def keys(self, keys):
        self._dataset = self._dataset.slice[:, keys]

    @property
    def mode(self):
        return self._dataset.mode

    @mode.setter
    def mode(self, mode):
        if mode is tuple:
            self._dataset = self._dataset.as_tuple()
        elif mode is dict:
            self._dataset = self._dataset.as_dict()
        elif mode is None:
            if len(self._dataset.keys) == 1:
                self._dataset = self._dataset.slice[:, 0]
            else:
                raise ValueError('Unary mode requires just one column')
        else:
            raise ValueError('Unknown mode: {}'.format(mode))

    def get_examples(self, indices, key_indices):
        return self._dataset.get_examples(indices, key_indices)
