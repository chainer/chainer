import numpy as np
import six
import unittest


from chainer.datasets import ConcatenatedDataset
from chainer import testing


@testing.parameterize(
    # basic usage
    {'datasets': (
        np.random.uniform(size=(5, 3, 48, 32)),
        np.random.uniform(size=(15, 3, 64, 48)),
    )},
    # more than two datasets
    {'datasets': (
        np.random.uniform(size=(5, 3, 48, 32)),
        np.random.uniform(size=(15, 3, 16, 48)),
        np.random.uniform(size=(20, 3, 5, 5)),
    )},
    # single dataset
    {'datasets': (
        np.random.uniform(size=(5, 3, 48, 32)),
    )},
    # no dataset
    {'datasets': ()},
    # some datasets are empty
    {'datasets': (
        np.random.uniform(size=(5, 3, 48, 32)),
        [],
        np.random.uniform(size=(20, 3, 5, 5)),
        [],
    )},
    # all datasets are empty
    {'datasets': ([], [], [])},
)
class TestConcatenatedDataset(unittest.TestCase):

    def setUp(self):
        self.concatenated_dataset = ConcatenatedDataset(*self.datasets)
        self.expected_dataset = [
            sample for dataset in self.datasets for sample in dataset]

    def test_concatenated_dataset(self):
        self.assertEqual(
            len(self.concatenated_dataset), len(self.expected_dataset))

        for i, expected in enumerate(self.expected_dataset):
            np.testing.assert_equal(self.concatenated_dataset[i], expected)

    def test_concatenated_dataset_slice(self):
        concatenated_slice = self.concatenated_dataset[1:8:2]
        expected_slice = self.concatenated_dataset[1:8:2]

        self.assertEqual(
            len(concatenated_slice), len(expected_slice))

        for concatenated, expected in six.moves.zip(
                concatenated_slice, expected_slice):
            np.testing.assert_equal(concatenated, expected)


testing.run_module(__name__, __file__)
