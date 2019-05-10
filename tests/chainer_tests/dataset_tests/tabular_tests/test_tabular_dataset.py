import numpy as np
import unittest

from chainer import testing
from chainer.dataset import TabularDataset


class DummyDataset(TabularDataset):

    def __init__(self, mode, get_examples):
        self._mode = mode
        self._get_examples = get_examples

    def __len__(self):
        return 10

    @property
    def keys(self):
        return ('a', 'b', 'c')

    @property
    def mode(self):
        return self._mode

    def get_examples(self, indices, key_indices):
        return self._get_examples(indices, key_indices)


@testing.parameterize(*testing.product({
    'integer': [int, np.int32],
    'mode': [tuple, dict],
}))
class TestTabularDataset(unittest.TestCase):

    def test_getitem_index(self):
        def get_examples(indices, key_indices):
            self.assertEqual(indices, [3])
            self.assertIsNone(key_indices)
            return [3], [1], [4]

        dataset = DummyDataset(self.mode, get_examples)
        output = dataset[self.integer(3)]
        if self.mode is tuple:
            self.assertEqual(output, (3, 1, 4))
        elif self.mode is dict:
            self.assertEqual(output, {'a': 3, 'b': 1, 'c': 4})


testing.run_module(__name__, __file__)
