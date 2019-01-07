import six
import typing as tp  # NOQA

import numpy

from chainer.utils import collections_abc


class Examples(collections_abc.Sequence):
    _is_tuple = False
    _is_dict = False

    def __init__(self, underlying):
        # type: (tp.Union[tp.Sequence, tp.Dict]) -> None
        # Note: Sequence includes both List and Tuple

        super(Examples, self).__init__()

        self._underlying = underlying

        if isinstance(underlying, tuple):
            self._is_tuple = True
        elif isinstance(underlying, dict):
            self._is_dict = True

    def __getitem__(self, index):
        if self._is_tuple:
            ret = [array[index] for array in self._underlying]
            if isinstance(index, (slice, list, numpy.ndarray)):
                length = len(ret[0])
                return [tuple([array[i] for array in ret])
                        for i in six.moves.range(length)]
            else:
                return tuple(ret)
        elif self._is_dict:
            ret = {key: array[index]
                   for key, array in six.iteritems(self._underlying)}
            if isinstance(index, (slice, list, numpy.ndarray)):
                length = len(six.next(six.itervalues(ret)))
                return [
                    {key: batch[i] for key, batch in six.iteritems(ret)}
                    for i in six.moves.range(length)]
            else:
                return ret
        else:
            return self.underlying[index]

    def __len__(self) -> int:
        if self._is_tuple:
            return len(self._underlying[0])
        elif self._is_dict:
            return len(six.next(six.itervalues(self._underlying)))  # type:ignore # NOQA
        else:
            return len(self._underlying)

    @property
    def underlying(self):
        return self._underlying

    @property
    def is_tuple(self):
        return self._is_tuple

    @property
    def is_dict(self):
        return self._is_dict
