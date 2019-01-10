from abc import abstractmethod

import six
import typing as tp  # NOQA

import numpy

import chainer
from chainer import backend
from chainer import types  # NOQA
from chainer.backends import cuda
import chainerx


def sample_examples(datasets, indices=None, padding_spec=None):
    # type: (tp.Union[types.Dataset, types.Datasets], tp.Optional[tp.Union[slice, tp.List[int], numpy.ndarray]], tp.Optional[types.PaddingSpec]) -> "Examples" # NOQA
    if isinstance(datasets, tuple):
        return TupleDatasetExamples(datasets, indices, padding_spec)
    elif isinstance(datasets, dict):
        return DictDatasetExamples(datasets, indices, padding_spec)
    else:
        return SingleDatasetExamples(datasets, indices, padding_spec)


class Examples:
    """
    An immutable list of examples.
    """

    def __init__(self):
        super(Examples, self).__init__()

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def to_dataset(self, indices=None, device=None):
        # type: (tp.Optional[tp.Union[int, slice]], tp.Optional[backend.Device]) -> tp.Union[types.Dataset, types.Datasets] # NOQA
        """
        Return the examples as dataset(s).

        Args:
            indices (int or slice): Indices of examples.
            device (device specifier): A device to which each array is sent.
                If it is omitted, all arrays are left in their original
                devices. See :meth:`~chainer.dataset.convert.to_device` for
                more details.

        Returns:
            Dataset, a tuple of datasets, or a dictionary of datasets. The
            type depends on the type of each example in the batch.
        """
        raise NotImplementedError


class AbstractDatasetExamples(Examples):
    """
    An immutable list of examples which are sampled from one or more datasets.
    """

    def __init__(self, datasets, indices=None, padding_spec=None):
        # type: (tp.Union[types.Dataset, types.Datasets], tp.Optional[tp.Union[slice, tp.List[int], numpy.ndarray]], tp.Optional[types.PaddingSpec]) -> None # NOQA

        super(AbstractDatasetExamples, self).__init__()
        self._datasets = self._sample_datasets(datasets, indices, padding_spec)

    def to_dataset(self, indices=None, device=None):
        if device is None:
            return self._datasets
        else:
            return self._map_datasets(device.send, self._datasets, indices)

    @abstractmethod
    def _sample_datasets(self, datasets, indices, padding_spec):
        # type: (tp.Union[types.Dataset, types.Datasets], tp.Optional[tp.Union[slice, tp.List[int], numpy.ndarray]], tp.Optional[types.PaddingSpec]) -> types.Dataset # NOQA
        raise NotImplementedError

    @abstractmethod
    def _map_datasets(self, f, datasets, indices):
        # type: (tp.Callable[[types.Dataset], types.Dataset], tp.Union[types.Dataset, types.Datasets], tp.Optional[tp.Union[int, slice]]) -> types.Dataset # NOQA
        raise NotImplementedError


class SingleDatasetExamples(AbstractDatasetExamples):
    def __init__(self, datasets, indices=None, padding_spec=None):
        super(SingleDatasetExamples, self).__init__(
            datasets, indices, padding_spec)

    def __getitem__(self, index):
        return self._datasets[index]

    def __len__(self):
        return len(self._datasets)

    def _sample_datasets(self, datasets, indices, padding_spec):
        if indices is None:
            indices = six.moves.range(datasets)

        return _sample_with_padding(datasets, indices, padding_spec)

    def _map_datasets(self, f, datasets, indices):
        if indices is None:
            return f(datasets)
        else:
            return f(datasets[indices])


class TupleDatasetExamples(AbstractDatasetExamples):
    def __init__(self, datasets, indices=None, padding_spec=None):
        super(TupleDatasetExamples, self).__init__(
            datasets, indices, padding_spec)

    def __getitem__(self, index):
        ret = [dataset[index] for dataset in self._datasets]
        if isinstance(index, slice):
            length = len(ret[0])
            return [tuple(dataset[i] for dataset in ret)
                    for i in six.moves.range(length)]
        else:
            return tuple(ret)

    def __len__(self):
        return len(self._datasets[0])

    def _sample_datasets(self, datasets, indices, padding_spec):
        datasets_len = len(datasets)

        if indices is None:
            indices = six.moves.range(datasets[0])

        if not isinstance(padding_spec, tuple):
            tuple_paddings = [padding_spec] * datasets_len
        else:
            tuple_paddings = padding_spec

        return tuple(
            _sample_with_padding(datasets[i], indices, tuple_paddings[i])
            for i in six.moves.range(datasets_len))

    def _map_datasets(self, f, datasets, indices):
        if indices is None:
            return tuple(f(dataset) for dataset in datasets)
        else:
            return tuple(f(dataset[indices]) for dataset in datasets)


class DictDatasetExamples(AbstractDatasetExamples):
    def __init__(self, datasets, indices=None, padding_spec=None):
        super(DictDatasetExamples, self).__init__(
            datasets, indices, padding_spec)

    def __getitem__(self, index):
        ret = {key: array[index]
               for key, array in six.iteritems(self._datasets)}
        if isinstance(index, slice):
            length = len(six.next(six.itervalues(ret)))
            return [{key: batch[i] for key, batch in six.iteritems(ret)}
                    for i in six.moves.range(length)]
        else:
            return ret

    def __len__(self):
        return len(six.next(six.itervalues(self.datasets)))

    def _sample_datasets(self, datasets, indices, padding_spec):
        if indices is None:
            indices = six.moves.range(
                six.next(six.itervalues(self._datasets)))

        if not isinstance(padding_spec, dict):
            dict_paddings = {key: padding_spec for key in self._datasets}
        else:
            dict_paddings = padding_spec

        return {k: _sample_with_padding(dataset, indices, dict_paddings[k])
                for k, dataset in datasets.items()}

    def _map_datasets(self, f, datasets, indices):
        if indices is None:
            return {k: f(dataset) for k, dataset in datasets}
        else:
            return {k: f(dataset[indices]) for k, dataset in datasets}


def _sample_with_padding(dataset, indices, padding=None):
    # type: (types.Dataset, tp.Union[slice, tp.List[int], numpy.ndarray], tp.Optional[types.PaddingSpec]) -> types.Dataset # NOQA

    if padding is None:
        if isinstance(dataset, (
                numpy.ndarray,
                cuda.ndarray,
                chainerx.ndarray)):
            # the dataset supports array indexing
            # TODO(okapies): replace to take method
            return dataset[indices]

        else:
            # TODO(okapies): dataset should be ndarray or not?
            head = dataset[0]
            target = backend.get_device_from_array(head)
            with chainer.using_device(target):
                if isinstance(indices, slice):
                    return target.xp.asarray(dataset[indices])
                else:
                    return target.xp.asarray([dataset[i] for i in indices])

    else:
        head = dataset[0]
        shape = numpy.array(head.shape, dtype=int)
        for array in dataset[1:]:
            if numpy.any(shape != array.shape):
                numpy.maximum(shape, array.shape, shape)
        ret_shape = tuple(numpy.insert(shape, 0, len(dataset)))

        target = backend.get_device_from_array(head)
        with chainer.using_device(target):
            result = target.xp.full(ret_shape, padding, dtype=head.dtype)

            # fill the result with the sampled examples
            if isinstance(indices, slice):
                for i, src in enumerate(dataset[indices]):
                    slices = tuple(slice(dim) for dim in src.shape)
                    result[(i,) + slices] = src  # type: ignore
            else:
                for i in six.moves.range(len(dataset)):
                    src = dataset[indices[i]]
                    slices = tuple(slice(dim) for dim in src.shape)
                    result[(i,) + slices] = src  # type: ignore

            return result
