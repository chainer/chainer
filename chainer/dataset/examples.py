import six
import typing as tp  # NOQA

import numpy

from chainer.utils import collections_abc


class Examples(collections_abc.Sequence):
    _is_tuple = False
    _is_dict = False

    def __init__(self, datasets):
        # type: (tp.Union[tp.Sequence, tp.Dict]) -> None
        # Note: Sequence includes both List and Tuple

        super(Examples, self).__init__()

        self._datasets = datasets

        if isinstance(datasets, tuple):
            self._is_tuple = True
        elif isinstance(datasets, dict):
            self._is_dict = True

    def __getitem__(self, index):
        if self._is_tuple:
            ret = [array[index] for array in self._datasets]
            if isinstance(index, (slice, list, numpy.ndarray)):
                length = len(ret[0])
                return [tuple([array[i] for array in ret])
                        for i in six.moves.range(length)]
            else:
                return tuple(ret)
        elif self._is_dict:
            ret = {key: array[index]
                   for key, array in six.iteritems(self._datasets)}
            if isinstance(index, (slice, list, numpy.ndarray)):
                length = len(six.next(six.itervalues(ret)))
                return [
                    {key: batch[i] for key, batch in six.iteritems(ret)}
                    for i in six.moves.range(length)]
            else:
                return ret
        else:
            return self._datasets[index]

    def __len__(self):
        if self._is_tuple:
            return len(self._datasets[0])
        elif self._is_dict:
            return len(six.next(six.itervalues(self._datasets)))  # type:ignore # NOQA
        else:
            return len(self._datasets)

    @property
    def underlying_datasets(self):
        return self._datasets

    @property
    def is_tuple(self):
        return self._is_tuple

    @property
    def is_dict(self):
        return self._is_dict
