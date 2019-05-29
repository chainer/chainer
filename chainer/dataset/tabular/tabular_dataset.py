import six

import chainer
from chainer.dataset import dataset_mixin


class TabularDataset(dataset_mixin.DatasetMixin):
    """An abstract class that represents tabular dataset.

    This class represents a tabular dataset.
    In a tabular dataset, all examples have the same number of elements.
    For example, all examples of the dataset below have three elements
    (:obj:`a[i]`, :obj:`b[i]`, and :obj:`c[i]`).

    .. csv-table::
        :header: , a, b, c

        0, :obj:`a[0]`, :obj:`b[0]`, :obj:`c[0]`
        1, :obj:`a[1]`, :obj:`b[1]`, :obj:`c[1]`
        2, :obj:`a[2]`, :obj:`b[2]`, :obj:`c[2]`
        3, :obj:`a[3]`, :obj:`b[3]`, :obj:`c[3]`

    Since an example can be represented by both tuple and dict (
    :obj:`(a[i], b[i], c[i])` and :obj:`{'a': a[i], 'b': b[i], 'c': c[i]}`),
    this class uses :attr:`mode` to indicate which representation will be used.

    An inheritance should implement
    :meth:`__len__`, :attr:`keys`, :attr:`mode` and :meth:`get_examples`.

    >>> import numpy as np
    >>>
    >>> from chainer import dataset
    >>>
    >>> class MyDataset(dataset.TabularDataset):
    ...
    ...     def __len__(self):
    ...         return 4
    ...
    ...     @property
    ...     def keys(self):
    ...          return ('a', 'b', 'c')
    ...
    ...     @property
    ...     def mode(self):
    ...          return tuple
    ...
    ...     def get_examples(self, indices, key_indices):
    ...          data = np.arange(12).reshape((4, 3))
    ...          if indices is not None:
    ...              data = data[indices]
    ...          if key_indices is not None:
    ...              data = data[:, list(key_indices)]
    ...          return tuple(data.transpose())
    ...
    >>> dataset = MyDataset()
    >>> len(dataset)
    4
    >>> dataset.keys
    ('a', 'b', 'c')
    >>> dataset.as_tuple()[0]
    (0, 1, 2)
    >>> sorted(dataset.as_dict()[0].items())
    [('a', 0), ('b', 1), ('c', 2)]
    >>>
    >>> view = dataset.slice[[3, 2], ('c', 0)]
    >>> len(view)
    2
    >>> view.keys
    ('c', 'a')
    >>> view.as_tuple()[1]
    (8, 6)
    >>> sorted(view.as_dict()[1].items())
    [('a', 6), ('c', 8)]

    """

    def __len__(self):
        raise NotImplementedError

    @property
    def keys(self):
        """Names of columns.

        A tuple of strings that indicate the names of columns.
        """
        raise NotImplementedError

    @property
    def mode(self):
        """Mode of representation.

        This indicates the type of value returend
        by :meth:`fetch` and :meth:`__getitem__`.
        :class:`tuple` and :class:`dict` are supported.
        """
        raise NotImplementedError

    def get_examples(self, indices, key_indices):
        """Return a part of data.

        Args:
            indices (list of ints or slice): Indices of requested rows.
                If this argument is :obj:`None`, it indicates all rows.
            key_indices (tuple of ints): Indices of requested columns.
                If this argument is :obj:`None`, it indicates all columns.

        Returns:
            tuple of lists/arrays
        """
        raise NotImplementedError

    @property
    def slice(self):
        """Get a slice of dataset.

        Args:
           indices (list/array of ints/bools or slice): Requested rows.
           keys (tuple of ints/strs): Requested columns.

        Returns:
            A view of specifed range.
        """
        return chainer.dataset.tabular._slice._SliceHelper(self)

    def fetch(self):
        """Fetch data.

        This method fetches all data of the dataset/view.
        Note that this method returns a column-major data
        (i.e. :obj:`([a[0], ..., a[3]], ..., [c[0], ... c[3]])` or
        :obj:`{'a': [a[0], ..., a[3]], ..., 'c': [c[0], ..., c[3]]}`).

        Returns:
            If :attr:`mode` is :class:`tuple`,
            this method returns a tuple of lists/arrays.
            If :attr:`mode` is :class:`dict`,
            this method returns a dict of lists/arrays.
        """
        examples = self.get_examples(None, None)
        if self.mode is tuple:
            return examples
        elif self.mode is dict:
            return dict(six.moves.zip(self.keys, examples))

    def as_tuple(self):
        """Return a view with tuple mode.

        Returns:
            A view whose :attr:`mode` is :class:`tuple`.
        """
        return chainer.dataset.tabular._as_mode._AsTuple(self)

    def as_dict(self):
        """Return a view with dict mode.

        Returns:
            A view whose :attr:`mode` is :class:`dict`.
        """
        return chainer.dataset.tabular._as_mode._AsDict(self)

    def get_example(self, i):
        example = self.get_examples([i], None)
        example = tuple(col[0] for col in example)
        if self.mode is tuple:
            return example
        elif self.mode is dict:
            return dict(six.moves.zip(self.keys, example))
