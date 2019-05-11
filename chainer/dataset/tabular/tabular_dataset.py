import six

from chainer.dataset import DatasetMixin


class TabularDataset(DatasetMixin):
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

    >>> len(dataet)
    4
    >>> dataet.keys
    ('a', 'b', 'c')
    >>>
    >>> dataet = dataset.as_tuple()
    >>> dataset.mode
    <class 'tuple'>
    >>> dataset[0]
    (0, 1, 2)  # (a[0], b[0], c[0])
    >>>
    >>> dataet = dataset.as_dict()
    >>> dataset.mode
    <class 'dict'>
    >>> dataset[0]
    {'a': 0, 'b': 1, 'c': 2)  # {'a': a[0], 'b': b[0], 'c': c[0]}

    An inheritance should implement
    :meth:`__len__`, :attr:`keys`, :attr:`mode` and :meth:`get_examples`.
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

        >>> len(dataet)
        4
        >>> dataet.keys
        ('a', 'b', 'c')
        >>>
        >>> dataset = dataset.slice[[3, 2], ('c', 0)]
        >>> len(dataet)
        2
        >>> dataet.keys
        ('c', 'a')

        Args:
           indices (list/array of ints/bools or slice): Requested rows.
           keys (tuple of ints/strs): Requested columns.

        Returns:
            A view of specifed range.
        """
        from chainer.dataset.tabular.slice import SliceHelper
        return SliceHelper(self)

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
        from chainer.dataset.tabular.as_mode import AsTuple
        return AsTuple(self)

    def as_dict(self):
        """Return a view with dict mode.

        Returns:
            A view whose :attr:`mode` is :class:`dict`.
        """
        from chainer.dataset.tabular.as_mode import AsDict
        return AsDict(self)

    def concat(self, *datasets):
        """Stack datasets along rows.

        Args:
            datasets (iterable of :class:`TabularDataset`):
                Datasets to be concatenated.
                All datasets must have the same :attr:`keys`.

        Returns:
            A concatenated dataset.
        """
        from chainer.dataset.tabular.concat import Concat
        return Concat(self, *datasets)

    def join(self, *datasets):
        """Stack datasets along columns.

        Args:
            datasets (iterable of :class:`TabularDataset`):
                Datasets to be concatenated.
                All datasets must have the same length

        Returns:
            A joined dataset.
        """
        from chainer.dataset.tabular.join import Join
        return Join(self, *datasets)

    def get_example(self, i):
        example = self.get_examples([i], None)
        example = tuple(col[0] for col in example)
        if self.mode is tuple:
            return example
        elif self.mode is dict:
            return dict(six.moves.zip(self.keys, example))
