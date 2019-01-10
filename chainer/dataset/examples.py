from abc import abstractmethod

import six
import typing as tp  # NOQA

import numpy

import chainer
from chainer import backend
from chainer import types  # NOQA
from chainer.backends import cuda
from chainer.utils import collections_abc
import chainerx


def sample_examples(datasets, indices=None, padding_spec=None):
    # type: (tp.Union[types.Dataset, types.Datasets], tp.Optional[tp.Union[tp.Sequence[int], numpy.ndarray]], tp.Optional[types.PaddingSpec]) -> "Examples" # NOQA
    if isinstance(datasets, tuple):
        return TupleDatasetExamples(datasets, indices, padding_spec)
    elif isinstance(datasets, dict):
        return DictDatasetExamples(datasets, indices, padding_spec)
    else:
        return SingleDatasetExamples(datasets, indices, padding_spec)


class Examples(collections_abc.Sequence):
    """
    An immutable list of examples.
    """

    def __init__(self):
        super(Examples, self).__init__()

    @abstractmethod
    def to_dataset(self, device=None):
        # type: (tp.Optional[backend.Device]) -> tp.Union[types.Dataset, types.Datasets] # NOQA
        """
        Return the examples as dataset(s).

        Args:
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
        # type: (tp.Union[types.Dataset, types.Datasets], tp.Optional[tp.Union[tp.Sequence[int], numpy.ndarray]], tp.Optional[types.PaddingSpec]) -> None # NOQA

        super(AbstractDatasetExamples, self).__init__()
        self._datasets = self._sample_datasets(datasets, indices, padding_spec)

    def to_dataset(self, device=None):
        # type: (tp.Optional[backend.Device]) -> tp.Union[types.Dataset, types.Datasets] # NOQA
        if device is None:
            return self._datasets
        else:
            return self._map_datasets(device.send, self._datasets)

    @abstractmethod
    def _sample_datasets(self, datasets, indices, padding_spec):
        raise NotImplementedError

    @abstractmethod
    def _map_datasets(self, f, datasets):
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

    def _map_datasets(self, f, datasets):
        return f(datasets)


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

    def _map_datasets(self, f, datasets):
        return tuple(f(dataset) for dataset in datasets)


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

    def _map_datasets(self, f, datasets):
        return {k: f(dataset) for k, dataset in datasets}


def _sample_with_padding(dataset, indices, padding=None):
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
                return target.xp.asarray(
                    [dataset[i] for i in indices])

    else:
        head = dataset[0]
        shape = _calc_max_shape(dataset)

        target = backend.get_device_from_array(head)
        with chainer.using_device(target):
            result = target.xp.full(shape, padding, dtype=head.dtype)

            # fill the result with the sampled examples
            for i in six.moves.range(len(dataset)):
                src = dataset[indices[i]]
                slices = tuple(slice(dim) for dim in src.shape)
                result[(i,) + slices] = src  # type: ignore

            return result


def _calc_max_shape(dataset):
    shape = numpy.array(dataset[0].shape, dtype=int)
    for array in dataset[1:]:
        if numpy.any(shape != array.shape):
            numpy.maximum(shape, array.shape, shape)
    return tuple(numpy.insert(shape, 0, len(dataset)))
