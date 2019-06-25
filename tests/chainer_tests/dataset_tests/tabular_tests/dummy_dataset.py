import numpy as np

import chainer
from chainer import testing


class DummyDataset(chainer.dataset.TabularDataset):

    def __init__(
            self, size=10, keys=('a', 'b', 'c'), mode=tuple,
            return_array=False, callback=None):
        self._keys = keys
        self._mode = mode
        self._return_array = return_array
        self._callback = callback

        self.data = np.random.uniform(size=(len(keys), size))

    def __len__(self):
        return self.data.shape[1]

    @property
    def keys(self):
        return self._keys

    @property
    def mode(self):
        return self._mode

    def get_examples(self, indices, key_indices):
        if self._callback:
            self._callback(indices, key_indices)

        data = self.data
        if indices is not None:
            data = data[:, indices]
        if key_indices is not None:
            data = data[list(key_indices)]

        if self._return_array:
            return tuple(data)
        else:
            return tuple(list(d) for d in data)


# tests/chainer_tests/test_runnable.py
testing.run_module(__name__, __file__)
