import bisect
import six
import typing as tp  # NOQA

import numpy

import chainer
from chainer import backend
from chainer import types  # NOQA
from chainer.backends import cuda
from chainer.dataset import convert
import chainerx


_ndarray_types = (
    numpy.ndarray,
    cuda.ndarray,
    chainerx.ndarray,
)


# only for internal use
if types.TYPE_CHECKING:
    _NdArrays = tp.Union[
        tp.Tuple[types.NdArray],
        tp.Dict[tp.Any, types.NdArray],
    ]


def sample_from_dataset(dataset, indices=None, order=None, padding_spec=None):
    # type: (tp.Union[types.Dataset, types.Datasets], tp.Optional[tp.Union[slice, tp.List[int], numpy.ndarray]], tp.Optional[tp.Sequence[int]], tp.Optional[types.PaddingSpec]) -> "Examples" # NOQA
    """
    Sample from the specified dataset(s).

    This function is useful to implement :meth:`get_examples` method in
    BatchableDatasetMixin.

    Note: Currently *indices* does not support numpy's fancy indexing;
    It only accepts 1-D slice, list and numpy.ndarray.

    Args:
        dataset (list, tuple of lists or dict of lists): Dataset(s).
        indices (None, slice, list or numpy.ndarray): Indices of dataset(s).
        order (sequence of ints): Permutation of indexes in the base dataset.
            If this is ``None``, then the ascending order of indexes is used.
        padding_spec: Scalar value for extra elements. If this is None
            (default), an error is raised on shape mismatch. Otherwise,
            an array of minimum dimensionalities that can accommodate
            all arrays is created, and elements outside of the examples
            are padded by this value.

    Returns:
        Examples
    """
    if isinstance(dataset, tuple):
        return TupleDatasetExamples(dataset, indices, order, padding_spec)  # type: ignore # NOQA
    elif isinstance(dataset, dict):
        return DictDatasetExamples(dataset, indices, order, padding_spec)  # type: ignore # NOQA
    else:
        return SingleDatasetExamples(dataset, indices, order, padding_spec)  # type: ignore # NOQA


class Examples:
    """
    A Sequence-like collection of examples.

    If you want a customized Examples, you just need to implement
    :meth:`to_dataset` method for :class:`~chainer.training.Updater` or
    :class:`~chainer.training.extensions.Evaluator` which supports Examples,
    but we also recommend you to implement :meth:`__len__` and
    :meth:`__getitem__` to keep backward compatibility.

    Note that it has same methods of :class:`~collections.abc.Sequence` but
    not an actual instance of Sequence to avoid the overhead of isinstance().
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

    def to_dataset(self, device_spec=None, indices=None, func=None):
        # type: (tp.Optional[tp.Union[int, types.DeviceSpec]], tp.Optional[tp.Union[int, slice]], tp.Callable[[tp.Union[types.Dataset, types.Datasets]], tp.Union[types.Dataset, types.Datasets]]) -> tp.Union[types.Dataset, types.Datasets] # NOQA
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
            func (callable): A function which is applied to the dataset(s)
                after sent to the specified device

        Returns:
            Dataset, a tuple of datasets, or a dictionary of datasets. The
            type depends on the type of each example in the batch.
        """
        raise NotImplementedError


class SingleDatasetExamples(Examples):
    def __init__(self, dataset, indices=None, order=None, padding_spec=None):
        # type: (types.Dataset, tp.Optional[tp.Union[slice, tp.List[int], numpy.ndarray]], tp.Optional[tp.Sequence[int]], tp.Optional[types.PaddingSpec]) -> None # NOQA

        super(SingleDatasetExamples, self).__init__()
        self._dataset = _sample_with_padding(dataset, indices, order, padding_spec)  # type: types.NdArray # NOQA

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        return self._dataset[index]

    def to_dataset(self, device_spec=None, indices=None, func=None):
        device = convert._get_device(device_spec)
        send = device.send if device is not None else _identity

        if indices is None:
            ret = send(self._dataset)
        else:
            ret = send(self._dataset[indices])

        return ret if func is None else func(ret)


class TupleDatasetExamples(Examples):
    def __init__(self, dataset, indices=None, order=None, padding_spec=None):
        # type: (tp.Tuple[types.Dataset, ...], tp.Optional[tp.Union[slice, tp.List[int], numpy.ndarray]], tp.Optional[tp.Sequence[int]], tp.Optional[types.PaddingSpec]) -> None # NOQA

        super(TupleDatasetExamples, self).__init__()

        datasets_len = len(dataset)

        if not isinstance(padding_spec, tuple):
            tuple_paddings = [padding_spec] * datasets_len
        else:
            tuple_paddings = padding_spec  # type: ignore

        self._dataset = tuple([
            _sample_with_padding(dataset[i], indices, order, tuple_paddings[i])
            for i in six.moves.range(datasets_len)])  # type: tp.Tuple[types.NdArray, ...] # NOQA

    def __len__(self):
        return len(self._dataset[0])

    def __getitem__(self, index):
        ret = [dataset[index] for dataset in self._dataset]
        if isinstance(index, slice):
            length = len(ret[0])
            return [tuple([dataset[i] for dataset in ret])
                    for i in six.moves.range(length)]
        else:
            return tuple(ret)

    def to_dataset(self, device_spec=None, indices=None, func=None):
        device = convert._get_device(device_spec)
        send = device.send if device is not None else _identity

        if indices is None:
            ret = tuple([send(d) for d in self._dataset])
        else:
            ret = tuple([send(d[indices]) for d in self._dataset])

        return ret if func is None else func(*ret)


class DictDatasetExamples(Examples):
    def __init__(self, dataset, indices=None, order=None, padding_spec=None):
        # type: (tp.Mapping[tp.Any, types.Dataset], tp.Optional[tp.Union[slice, tp.List[int], numpy.ndarray]], tp.Optional[tp.Sequence[int]], tp.Optional[types.PaddingSpec]) -> None # NOQA

        super(DictDatasetExamples, self).__init__()

        if not isinstance(padding_spec, dict):
            dict_paddings = {key: padding_spec for key in dataset}
        else:
            dict_paddings = padding_spec

        self._dataset = {
            k: _sample_with_padding(dataset, indices, order, dict_paddings[k])
            for k, dataset in six.iteritems(dataset)}  # type: tp.Mapping[tp.Any, types.NdArray] # NOQA

    def __len__(self):
        return len(six.next(six.itervalues(self._dataset)))

    def __getitem__(self, index):
        ret = {key: array[index]
               for key, array in six.iteritems(self._dataset)}
        if isinstance(index, slice):
            length = len(six.next(six.itervalues(ret)))
            return [{key: batch[i] for key, batch in six.iteritems(ret)}
                    for i in six.moves.range(length)]
        else:
            return ret

    def to_dataset(self, device_spec=None, indices=None, func=None):
        device = convert._get_device(device_spec)
        send = device.send if device is not None else _identity

        if indices is None:
            ret = {
                k: send(d) for k, d in six.iteritems(self._dataset)}
        else:
            ret = {
                k: send(d[indices]) for k, d in six.iteritems(self._dataset)}

        return ret if func is None else func(**ret)


class ConcatenatedExamples(Examples):
    def __init__(self, datasets, indices=None, padding_spec=None):
        # type: (tp.Sequence[types.Dataset], tp.Optional[tp.Union[slice, tp.List[int], numpy.ndarray]], tp.Optional[types.PaddingSpec]) -> None # NOQA
        super(ConcatenatedExamples, self).__init__()

        ds_idxs = self._to_dataset_indices(datasets, indices)

        # TODO(okapies): improve performance
        head_ds, head_idx = ds_idxs[0]
        head = datasets[head_ds][head_idx]
        if isinstance(head, tuple):
            tuple_ds = tuple([[] for _ in six.moves.xrange(len(head))])  # type: tp.Tuple[tp.List[tp.Any], ...] # NOQA
            for n, i in ds_idxs:
                for k, e in enumerate(datasets[n][i]):
                    tuple_ds[k].append(e)
            self._examples =\
                TupleDatasetExamples(tuple_ds, None, None, padding_spec)  # type: Examples # NOQA
        elif isinstance(head, dict):
            dict_ds = {key: [] for key in head}  # type: tp.Mapping[tp.Any, tp.List[tp.Any]] # NOQA
            for n, i in ds_idxs:
                for k, e in six.iteritems(datasets[n][i]):
                    dict_ds[k].append(e)
            self._examples =\
                DictDatasetExamples(dict_ds, None, None, padding_spec)
        else:
            single_ds = []  # type: tp.List[tp.Any]
            for n, i in ds_idxs:
                single_ds.append(datasets[n][i])
            self._examples =\
                SingleDatasetExamples(single_ds, None, None, padding_spec)

    @staticmethod
    def _to_dataset_indices(datasets, indices):
        # type: (tp.Sequence[types.Dataset], tp.Optional[tp.Union[slice, tp.List[int], numpy.ndarray]]) -> tp.Sequence[tp.Tuple[int, int]] # NOQA

        if indices is None:
            n_datasets = sum(
                len(dataset) for dataset in datasets)  # type: ignore
            idxs = six.moves.xrange(n_datasets)  # type: tp.Sequence[int]
        elif isinstance(indices, slice):
            n_datasets = sum(
                len(dataset) for dataset in datasets)  # type: ignore
            idxs = six.moves.xrange(n_datasets)[indices]
        else:
            idxs = indices

        # calculate offsets of each dataset
        offset = 0
        offsets = []  # type: tp.List[int]
        for dataset in datasets:
            offsets.append(offset)
            offset += len(dataset)

        # e.g. [0, 1, 3, ...] -> [(0, 0), (1, 0), (3, 1), ...]
        ret = []
        for i in idxs:
            if i < 0:
                raise IndexError

            # bisect_right(offsets, i) >= 1 (because offsets[0] is always 0)
            ds_idx = bisect.bisect_right(offsets, i) - 1
            ret.append((ds_idx, i - offsets[ds_idx]))

        return ret

    def __len__(self):
        # type: () -> int
        return len(self._examples)

    def __getitem__(self, index):
        return self._examples[index]

    def to_dataset(self, device_spec=None, indices=None, func=None):
        return self._examples.to_dataset(device_spec, indices, func)


def _identity(a):
    # type: (tp.Any) -> tp.Any
    return a


def _sample_with_padding(dataset, indices, order=None, padding=None):
    # type: (types.Dataset, tp.Optional[tp.Union[slice, tp.List[int], numpy.ndarray]], tp.Optional[tp.Sequence[int]], tp.Optional[types.PaddingSpec]) -> types.NdArray # NOQA

    if order is None:
        idxs = indices
    elif indices is None:
        idxs = [order[i] for i in six.moves.xrange(len(dataset))]
    elif isinstance(indices, slice):
        idxs = [order[i] for i in six.moves.xrange(len(dataset))[indices]]
    else:
        idxs = [order[i] for i in indices]

    if padding is None:
        if isinstance(dataset, _ndarray_types):
            if idxs is None:
                return dataset
            else:
                # the dataset supports indexing by list and numpy.ndarray
                # TODO(okapies): replace to take method
                return dataset[idxs]

        else:
            # convert types implementing dataset protocol (including list)
            # to ndarray
            return _dataset_to_ndarray(dataset, idxs)

    else:
        return _create_padded_examples(dataset, idxs, padding)


def _dataset_to_ndarray(dataset, indices=None):
    # type: (types.Dataset, tp.Optional[tp.Union[slice, tp.List[int], numpy.ndarray]]) -> types.NdArray # NOQA
    if indices is None:
        head = dataset[0]
        target = backend.get_device_from_array(head)
        with chainer.using_device(target):
            if isinstance(dataset, list):
                return target.xp.asarray(dataset)
            else:
                # assume dataset only have __getitem__ and __len__
                return target.xp.asarray(
                    [dataset[i]
                     for i in six.moves.xrange(len(dataset))])

    elif isinstance(indices, slice):
        head = dataset[indices.indices(len(dataset))[0]]
        target = backend.get_device_from_array(head)
        examples = dataset[indices]
        with chainer.using_device(target):
            if isinstance(examples, list):
                return target.xp.asarray(examples)
            else:
                # assume dataset only have __getitem__ and __len__
                return target.xp.asarray(
                    [examples[i]
                     for i in six.moves.xrange(len(examples))])

    else:
        head = dataset[indices[0]]
        target = backend.get_device_from_array(head)
        with chainer.using_device(target):
            return target.xp.asarray([dataset[i] for i in indices])


def _create_padded_examples(dataset, indices, padding):
    # type: (types.Dataset, tp.Optional[tp.Union[slice, tp.List[int], numpy.ndarray]], types.PaddingSpec) -> types.NdArray # NOQA
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
            slices = tuple([slice(dim) for dim in src.shape])
            result[(i,) + slices] = src  # type: ignore

        return result
