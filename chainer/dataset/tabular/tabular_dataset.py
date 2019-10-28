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
    If there is only one column, an example also can be represented by a value
    (:obj:`a[i]`). In this case, :attr:`mode` is :obj:`None`.

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
    >>> dataset.astuple()[0]
    (0, 1, 2)
    >>> sorted(dataset.asdict()[0].items())
    [('a', 0), ('b', 1), ('c', 2)]
    >>>
    >>> view = dataset.slice[[3, 2], ('c', 0)]
    >>> len(view)
    2
    >>> view.keys
    ('c', 'a')
    >>> view.astuple()[1]
    (8, 6)
    >>> sorted(view.asdict()[1].items())
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

        This indicates the type of value returned
        by :meth:`fetch` and :meth:`__getitem__`.
        :class:`tuple`, :class:`dict`, and :obj:`None` are supported.
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
           keys (tuple of ints/strs or int or str): Requested columns.

        Returns:
            A view of specified range.
        """
        return chainer.dataset.tabular._slice._SliceHelper(self)

    def fetch(self):
        """Fetch data.

        This method fetches all data of the dataset/view.
        Note that this method returns a column-major data
        (i.e. :obj:`([a[0], ..., a[3]], ..., [c[0], ... c[3]])`,
        :obj:`{'a': [a[0], ..., a[3]], ..., 'c': [c[0], ..., c[3]]}`, or
        :obj:`[a[0], ..., a[3]]`).

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
        elif self.mode is None:
            return examples[0]

    def convert(self, data):
        """Convert fetched data.

        This method takes data fetched by :meth:`fetch` and
        pre-process them before passing them to models.
        The default behaviour is converting each column into an ndarray.
        This behaviour can be overridden by :meth:`with_converter`.
        If the dataset is constructed by :meth:`concat` or :meth:`join`,
        the converter of the first dataset is used.

        Args:
            data (tuple or dict): Data from :meth:`fetch`.

        Returns:
            A tuple or dict.
            Each value is an ndarray.
        """
        if isinstance(data, tuple):
            return tuple(_as_array(d) for d in data)
        elif isinstance(data, dict):
            return {k: _as_array(v) for k, v in data.items()}
        else:
            return _as_array(data)

    def astuple(self):
        """Return a view with tuple mode.

        Returns:
            A view whose :attr:`mode` is :class:`tuple`.
        """
        return chainer.dataset.tabular._asmode._Astuple(self)

    def asdict(self):
        """Return a view with dict mode.

        Returns:
            A view whose :attr:`mode` is :class:`dict`.
        """
        return chainer.dataset.tabular._asmode._Asdict(self)

    def concat(self, *datasets):
        """Stack datasets along rows.

        Args:
            datasets (iterable of :class:`TabularDataset`):
                Datasets to be concatenated.
                All datasets must have the same :attr:`keys`.

        Returns:
            A concatenated dataset.
        """
        return chainer.dataset.tabular._concat._Concat(self, *datasets)

    def join(self, *datasets):
        """Stack datasets along columns.

        Args:
            datasets (iterable of :class:`TabularDataset`):
                Datasets to be concatenated.
                All datasets must have the same length

        Returns:
            A joined dataset.
        """
        return chainer.dataset.tabular._join._Join(self, *datasets)

    def transform(self, keys, transform):
        """Apply a transform to each example.

        Args:
            keys (tuple of strs): The keys of transformed examples.
            transform (callable): A callable that takes an example
                and returns transformed example. :attr:`mode` of
                transformed dataset is determined by the transformed
                examples.

        Returns:
            A transfromed dataset.
        """
        return chainer.dataset.tabular._transform._Transform(
            self, keys, transform)

    def transform_batch(self, keys, transform_batch):
        """Apply a transform to examples.

        Args:
            keys (tuple of strs): The keys of transformed examples.
            transform_batch (callable): A callable that takes examples
                and returns transformed examples. :attr:`mode` of
                transformed dataset is determined by the transformed
                examples.

        Returns:
            A transfromed dataset.
        """
        return chainer.dataset.tabular._transform._TransformBatch(
            self, keys, transform_batch)

    def with_converter(self, converter):
        """Override the behaviour of :meth:`convert`.

        This method overrides :meth:`convert`.

        Args:
            converter (callable): A new converter.

        Returns:
            A dataset with the new converter.
        """

        return chainer.dataset.tabular._with_converter._WithConverter(
            self, converter)

    def get_example(self, i):
        example = self.get_examples([i], None)
        example = tuple(col[0] for col in example)
        if self.mode is tuple:
            return example
        elif self.mode is dict:
            return dict(six.moves.zip(self.keys, example))
        elif self.mode is None:
            return example[0]

    def __iter__(self):
        return (self.get_example(i) for i in six.moves.range(len(self)))


def _as_array(data):
    if isinstance(data, chainer.get_array_types()):
        return data
    else:
        device = chainer.backend.get_device_from_array(data[0])
        with chainer.using_device(device):
            return device.xp.asarray(data)
