import unittest

import numpy as np

from chainer.datasets import ConcatenatedDataset
from chainer import testing


class TestConcatenatedDataset(unittest.TestCase):

    def test_concatenated_dataset(self):
        dataset0 = np.random.uniform(size=(5, 3, 32, 32))
        dataset1 = np.random.uniform(size=(15, 3, 32, 32))
        concatenated_dataset = ConcatenatedDataset(dataset0, dataset1)

        self.assertEqual(len(concatenated_dataset), 20)

        np.testing.assert_equal(concatenated_dataset[0], dataset0[0])
        np.testing.assert_equal(concatenated_dataset[4], dataset0[4])
        np.testing.assert_equal(concatenated_dataset[5], dataset1[0])
        np.testing.assert_equal(concatenated_dataset[8], dataset1[3])

    def test_concatenated_dataset_slice(self):
        dataset0 = np.random.uniform(size=(5, 3, 32, 32))
        dataset1 = np.random.uniform(size=(15, 3, 32, 32))
        concatenated_dataset = ConcatenatedDataset(dataset0, dataset1)

        self.assertEqual(len(concatenated_dataset), 20)

        out = concatenated_dataset[1:8:2]
        self.assertEqual(len(out), 4)
        np.testing.assert_equal(out[0], dataset0[1])
        np.testing.assert_equal(out[1], dataset0[3])
        np.testing.assert_equal(out[2], dataset1[0])
        np.testing.assert_equal(out[3], dataset1[2])


testing.run_module(__name__, __file__)
