import six
import typing as tp  # NOQA

import numpy

import chainer
from chainer import backend
from chainer import types  # NOQA
from chainer.backends import cuda
from chainer.utils import collections_abc
import chainerx


class SampledExamples(collections_abc.Sequence):
    """
    An immutable list of examples which are sampled from one or more datasets.
    """

    _is_tuple = False
    _is_dict = False

    def __init__(self, datasets, indices=None, padding_spec=None, eager=False):
        # type: (tp.Union[types.Dataset, types.Datasets], tp.Optional[tp.Sequence[int]], tp.Optional[types.PaddingSpec], bool) -> None # NOQA
        # Note: Sequence includes both List and Tuple

        super(SampledExamples, self).__init__()

        self._datasets = datasets

        if isinstance(datasets, tuple):
            self._is_tuple = True
        elif isinstance(datasets, dict):
            self._is_dict = True

        if indices is not None:
            self._indices = indices
        else:
            # It should be run after all other fields are initialized
            self._indices = six.moves.range(self._length_of_source_datasets())

    def _length_of_source_datasets(self):
        if self._is_tuple:
            return len(self._datasets[0])
        elif self._is_dict:
            return len(six.next(six.itervalues(self._datasets)))  # type:ignore # NOQA
        else:
            return len(self._datasets)

    def __getitem__(self, index):
        if self._is_tuple:
            ret = [dataset[self._indices[index]] for dataset in self._datasets]
            if isinstance(index, slice):
                length = len(ret[0])
                return [tuple(dataset[i] for dataset in ret)
                        for i in six.moves.range(length)]
            else:
                return tuple(ret)
        elif self._is_dict:
            ret = {key: array[self._indices[index]]
                   for key, array in six.iteritems(self._datasets)}
            if isinstance(index, slice):
                length = len(six.next(six.itervalues(ret)))
                return [{key: batch[i] for key, batch in six.iteritems(ret)}
                        for i in six.moves.range(length)]
            else:
                return ret
        else:
            return self._datasets[self._indices[index]]

    def __len__(self):
        return len(self._indices)

    def datasets(self, f=None, padding_spec=None):
        # type: (tp.Callable[[types.Dataset], types.Dataset], tp.Optional[types.PaddingSpec]) -> tp.Union[types.Dataset, types.Datasets] # NOQA

        if f is None:
            f = _identity

        if self._is_tuple:
            datasets_len = len(self._datasets)  # type: ignore
            if not isinstance(padding_spec, tuple):
                tuple_paddings = [padding_spec] * datasets_len  # type: ignore # NOQA
            else:
                tuple_paddings = padding_spec  # type: ignore # NOQA

            return tuple(
                f(self._run_sample(self._datasets[i], tuple_paddings[i]))  # type: ignore # NOQA
                for i in six.moves.range(datasets_len))

        elif self._is_dict:
            if not isinstance(padding_spec, dict):
                dict_paddings = {key: padding_spec for key in self._datasets}  # type: ignore # NOQA
            else:
                dict_paddings = padding_spec

            return {k: f(self._run_sample(dataset, dict_paddings[k]))
                    for k, dataset in self._datasets.items()}  # type: ignore # NOQA

        else:
            return f(self._run_sample(self._datasets, padding_spec))  # type: ignore # NOQA

    def _run_sample(self, dataset, padding=None):
        # type: (types.Dataset, tp.Optional[types.ScalarValue]) -> types.Dataset # NOQA

        if padding is None:
            if isinstance(dataset, (
                    numpy.ndarray,
                    cuda.ndarray,
                    chainerx.ndarray)):
                # the dataset supports array indexing
                return dataset[self._indices]

            else:
                # TODO(okapies): dataset should be ndarray or not?
                head = dataset[0]
                target = backend.get_device_from_array(head)
                with chainer.using_device(target):
                    return target.xp.asarray(
                        [dataset[i] for i in self._indices])

        else:
            head = dataset[0]
            shape = _calc_max_shape(dataset)

            target = backend.get_device_from_array(head)
            with chainer.using_device(target):
                result = target.xp.full(shape, padding, dtype=head.dtype)

                # fill the result with the sampled examples
                for i in six.moves.range(len(dataset)):
                    src = dataset[self._indices[i]]
                    slices = tuple(slice(dim) for dim in src.shape)
                    result[(i,) + slices] = src  # type: ignore

                return result

    def datasets_to(self, device, padding_spec=None):
        # type: (tp.Optional[backend.Device], tp.Optional[types.PaddingSpec]) -> tp.Union[types.Dataset, types.Datasets] # NOQA
        if device is None:
            return self.datasets(padding_spec=padding_spec)
        else:
            return self.datasets(device.send, padding_spec=padding_spec)

    def map_datasets(self, f, padding_spec=None):
        # type: (tp.Callable[[types.Dataset], types.Dataset], tp.Optional[types.PaddingSpec]) -> "SampledExamples" # NOQA
        return SampledExamples(
            self.datasets(f, padding_spec),
            six.moves.range(len(self._indices)))


def _identity(a):
    # type: (tp.Any) -> tp.Any
    return a


def _calc_max_shape(dataset):
    shape = numpy.array(dataset[0].shape, dtype=int)
    for array in dataset[1:]:
        if numpy.any(shape != array.shape):
            numpy.maximum(shape, array.shape, shape)
    return tuple(numpy.insert(shape, 0, len(dataset)))
