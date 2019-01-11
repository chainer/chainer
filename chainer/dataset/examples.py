import six
import typing as tp  # NOQA

import numpy

import chainer
from chainer import backend
from chainer import types  # NOQA
from chainer.backends import cuda
from chainer.dataset import convert
import chainerx


def sample_from_dataset(dataset, indices=None, padding_spec=None):
    # type: (tp.Union[types.Dataset, types.Datasets], tp.Optional[tp.Union[slice, tp.List[int], numpy.ndarray]], tp.Optional[types.PaddingSpec]) -> "Examples" # NOQA
    """
    Sample from the specified dataset(s).

    This function is useful to implement :meth:`get_examples` method in
    BatchableDatasetMixin.

    Note: Currently *indices* does not support numpy's fancy indexing;
    It only accepts 1-D slice, list and numpy.ndarray.

    Args:
        dataset (list, tuple of lists or dict of lists): Dataset(s).
        indices (None, slice, list or numpy.ndarray): Indices of dataset(s).
        padding_spec: Scalar value for extra elements. If this is None
            (default), an error is raised on shape mismatch. Otherwise,
            an array of minimum dimensionalities that can accommodate
            all arrays is created, and elements outside of the examples
            are padded by this value.

    Returns:
        Examples
    """
    if isinstance(dataset, tuple):
        return TupleDatasetExamples(dataset, indices, padding_spec)
    elif isinstance(dataset, dict):
        return DictDatasetExamples(dataset, indices, padding_spec)
    else:
        return SingleDatasetExamples(dataset, indices, padding_spec)


class Examples:
    """
    A Sequence-like collection of examples.

    If you want a customized Examples, you just need to implement
    :meth:`to_dataset` method for :class:`~chainer.training.Updater`
    which supports Examples, but we also recommend you to implement
    :meth:`__len__` and :meth:`__getitem__` to keep backward compatibility.

    Note that it has same methods of :class:`~collections.abc.Sequence` but
    not actually an instance of Sequence to avoid the overhead of isinstance().
    """

    def __init__(self):
        super(Examples, self).__init__()

    def __contains__(self, value):
        # type: (tp.Any) -> bool

        for e in self:  # self.__iter__
            if e is value or e == value:
                return True
        return False

    def __reversed__(self):
        # type: () -> tp.Iterator[tp.Any]

        for i in reversed(six.moves.xrange(len(self))):  # self.__len__
            yield self[i]  # self.__getitem__

    def __iter__(self):
        # type: () -> tp.Iterator[tp.Any]

        return (
            self[i]  # self.__getitem__
            for i in six.moves.xrange(len(self)))  # self.__len__

    def index(self, value, start=0, stop=None):
        # type: (tp.Any, int, int) -> int

        for i in six.moves.xrange(len(self))[slice(start, stop)]:  # __len__
            e = self[i]  # self.__getitem__
            if e is value or e == value:
                return i
        raise ValueError

    def count(self, value):
        # type: (tp.Any) -> int

        return sum(1 for e in self if e is value or e == value)  # __iter__

    def __len__(self):
        # type: () -> int
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def to_dataset(self, device_spec=None, indices=None):
        # type: (tp.Optional[tp.Union[int, types.DeviceSpec]], tp.Optional[tp.Union[int, slice]]) -> tp.Union[types.Dataset, types.Datasets] # NOQA
        """
        Return the examples as dataset(s).

        Args:
            device_spec (None or int or device specifier): A device to which
                examples are sent. If it is a negative integer, an array is
                sent to CPU. If it is a positive integer, an array is sent to
                GPU with the given ID. If it is``None``, an array is left in
                the original device. Also, any of device specifiers described
                at :class:`~chainer.backend.DeviceId` is accepted.
            indices (None or int or slice): Indices of examples. This method
                extracts the specified examples before sending them to the
                device.

        Returns:
            Dataset, a tuple of datasets, or a dictionary of datasets. The
            type depends on the type of each example in the batch.
        """
        raise NotImplementedError


class AbstractDatasetExamples(Examples):
    """
    A Sequence-like collection of examples sampled from one or more datasets.

    This is an abstract base class for Single, Tuple and DictDatasetExamples.
    """

    def __init__(self, dataset, indices=None, padding_spec=None):
        # type: (tp.Union[types.Dataset, types.Datasets], tp.Optional[tp.Union[slice, tp.List[int], numpy.ndarray]], tp.Optional[types.PaddingSpec]) -> None # NOQA

        super(AbstractDatasetExamples, self).__init__()
        self._dataset = self._do_sample(dataset, indices, padding_spec)

    def to_dataset(self, device_spec=None, indices=None):
        device = convert.resolve_device_spec(device_spec)
        if device is None:
            converter = _identity
        else:
            converter = device.send

        return self._map_datasets(converter, self._dataset, indices)

    def _do_sample(self, dataset, indices, padding_spec):
        # type: (tp.Union[types.Dataset, types.Datasets], tp.Optional[tp.Union[slice, tp.List[int], numpy.ndarray]], tp.Optional[types.PaddingSpec]) -> tp.Union[types.Dataset, types.Datasets] # NOQA
        raise NotImplementedError

    def _map_datasets(self, f, dataset, indices):
        # type: (tp.Callable[[types.Dataset], types.Dataset], tp.Union[types.Dataset, types.Datasets], tp.Optional[tp.Union[int, slice]]) -> tp.Union[types.Dataset, types.Datasets] # NOQA
        raise NotImplementedError


class SingleDatasetExamples(AbstractDatasetExamples):
    def __init__(self, dataset, indices=None, padding_spec=None):
        super(SingleDatasetExamples, self).__init__(
            dataset, indices, padding_spec)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        return self._dataset[index]

    def _do_sample(self, dataset, indices, padding_spec):
        return _sample_with_padding(dataset, indices, padding_spec)

    def _map_datasets(self, f, dataset, indices):
        if indices is None:
            return f(dataset)
        else:
            return f(dataset[indices])


class TupleDatasetExamples(AbstractDatasetExamples):
    def __init__(self, dataset, indices=None, padding_spec=None):
        super(TupleDatasetExamples, self).__init__(
            dataset, indices, padding_spec)

    def __getitem__(self, index):
        ret = [dataset[index] for dataset in self._dataset]
        if isinstance(index, slice):
            length = len(ret[0])
            return [tuple(dataset[i] for dataset in ret)
                    for i in six.moves.range(length)]
        else:
            return tuple(ret)

    def __len__(self):
        return len(self._dataset[0])

    def _do_sample(self, dataset, indices, padding_spec):
        datasets_len = len(dataset)

        if not isinstance(padding_spec, tuple):
            tuple_paddings = [padding_spec] * datasets_len
        else:
            tuple_paddings = padding_spec

        return tuple(
            _sample_with_padding(dataset[i], indices, tuple_paddings[i])
            for i in six.moves.range(datasets_len))

    def _map_datasets(self, f, dataset, indices):
        if indices is None:
            return tuple(f(dataset) for dataset in dataset)
        else:
            return tuple(f(dataset[indices]) for dataset in dataset)


class DictDatasetExamples(AbstractDatasetExamples):
    def __init__(self, dataset, indices=None, padding_spec=None):
        super(DictDatasetExamples, self).__init__(
            dataset, indices, padding_spec)

    def __getitem__(self, index):
        ret = {key: array[index]
               for key, array in six.iteritems(self._dataset)}
        if isinstance(index, slice):
            length = len(six.next(six.itervalues(ret)))
            return [{key: batch[i] for key, batch in six.iteritems(ret)}
                    for i in six.moves.range(length)]
        else:
            return ret

    def __len__(self):
        return len(six.next(six.itervalues(self._dataset)))

    def _do_sample(self, dataset, indices, padding_spec):
        if not isinstance(padding_spec, dict):
            dict_paddings = {key: padding_spec for key in self._dataset}
        else:
            dict_paddings = padding_spec

        return {k: _sample_with_padding(dataset, indices, dict_paddings[k])
                for k, dataset in dataset.items()}

    def _map_datasets(self, f, dataset, indices):
        if indices is None:
            return {k: f(dataset) for k, dataset in dataset}
        else:
            return {k: f(dataset[indices]) for k, dataset in dataset}


def _identity(a):
    # type: (tp.Any) -> tp.Any
    return a


def _sample_with_padding(dataset, indices, padding=None):
    # type: (types.Dataset, tp.Optional[tp.Union[slice, tp.List[int], numpy.ndarray]], tp.Optional[types.PaddingSpec]) -> types.Dataset # NOQA

    if padding is None:
        if isinstance(dataset, (
                numpy.ndarray,
                cuda.ndarray,
                chainerx.ndarray)):
            if indices is None:
                return dataset
            else:
                # the dataset supports indexing by list and numpy.ndarray
                # TODO(okapies): replace to take method
                return dataset[indices]

        else:
            # TODO(okapies): dataset should be ndarray or not?
            if indices is None:
                head = dataset[0]
                target = backend.get_device_from_array(head)
                with chainer.using_device(target):
                    return target.xp.asarray(dataset)
            elif isinstance(indices, slice):
                head = dataset[indices.indices(len(dataset))[0]]
                target = backend.get_device_from_array(head)
                with chainer.using_device(target):
                    return target.xp.asarray(dataset[indices])
            else:
                head = dataset[indices[0]]
                target = backend.get_device_from_array(head)
                with chainer.using_device(target):
                    return target.xp.asarray([dataset[i] for i in indices])

    else:
        return _create_padded_examples(dataset, indices, padding)


def _create_padded_examples(dataset, indices, padding):
    # type: (types.Dataset, tp.Optional[tp.Union[slice, tp.List[int], numpy.ndarray]], types.PaddingSpec) -> types.Dataset # NOQA
    if indices is None:
        idxs = six.moves.xrange(len(dataset))  # type: tp.Sequence
    elif isinstance(indices, slice):
        idxs = six.moves.xrange(len(dataset))[indices]
    else:
        idxs = indices

    # TODO(okapies): how to calculate a shape in parallel updater?
    head = dataset[idxs[0]]
    shape = numpy.array(head.shape, dtype=int)
    for i in idxs[1:]:
        array = dataset[i]
        if numpy.any(shape != array.shape):
            numpy.maximum(shape, array.shape, shape)
    total_shape = tuple(numpy.insert(shape, 0, len(idxs)))

    target = backend.get_device_from_array(head)
    with chainer.using_device(target):
        result = target.xp.full(total_shape, padding, dtype=head.dtype)

        # fill the result with the sampled examples
        for i, j in enumerate(idxs):
            src = dataset[j]
            slices = tuple(slice(dim) for dim in src.shape)
            result[(i,) + slices] = src  # type: ignore

        return result
