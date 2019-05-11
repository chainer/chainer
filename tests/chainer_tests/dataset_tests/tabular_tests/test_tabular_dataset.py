import numpy as np
import unittest

from chainer import testing
from chainer.dataset import TabularDataset


class DummyDataset(TabularDataset):

    def __init__(self, mode, return_array=False, callback=None):
        self._mode = mode
        self._return_array = return_array
        self._callback = callback

        self.data = np.random.uniform(size=(3, 10))

    def __len__(self):
        return self.data.shape[1]

    @property
    def keys(self):
        return ('a', 'b', 'c')

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


@testing.parameterize(*testing.product({
    'mode': [tuple, dict],
    'return_array': [True, False],
}))
class TestTabularDataset(unittest.TestCase):

    def test_fetch(self):
        def callback(indices, key_indices):
            self.assertIsNone(indices)
            self.assertIsNone(key_indices)

        dataset = DummyDataset(
            mode=self.mode, return_array=self.return_array, callback=callback)
        output = dataset.fetch()

        if self.mode is tuple:
            expected = tuple(dataset.data)
        elif self.mode is dict:
            expected = dict(zip(('a', 'b', 'c'), dataset.data))
        np.testing.assert_equal(output, expected)

        if self.mode is dict:
            output = output.values()
        for out in output:
            if self.return_array:
                self.assertIsInstance(out, np.ndarray)
            else:
                self.assertIsInstance(out, list)

    def test_get_example(self):
        def callback(indices, key_indices):
            self.assertEqual(indices, [3])
            self.assertIsNone(key_indices)

        dataset = DummyDataset(
            mode=self.mode, return_array=self.return_array, callback=callback)

        if self.mode is tuple:
            expected = tuple(dataset.data[:, 3])
        elif self.mode is dict:
            expected = dict(zip(('a', 'b', 'c'), dataset.data[:, 3]))

        self.assertEqual(dataset.get_example(3), expected)


testing.run_module(__name__, __file__)
