import unittest

import numpy as np

from chainermn.datasets import create_empty_dataset


class TestEmptyDataset(unittest.TestCase):

    def setUp(self):
        pass

    def check_create_empty_dataset(self, original_dataset):
        empty_dataset = create_empty_dataset(original_dataset)
        self.assertEqual(len(original_dataset), len(empty_dataset))
        for i in range(len(original_dataset)):
            self.assertEqual((), empty_dataset[i])

    def test_scatter_dataset(self):
        n = 10

        self.check_create_empty_dataset([])
        self.check_create_empty_dataset([0])
        self.check_create_empty_dataset(list(range(n)))
        self.check_create_empty_dataset(list(range(n * 5 - 1)))

        self.check_create_empty_dataset(np.array([]))
        self.check_create_empty_dataset(np.array([0]))
        self.check_create_empty_dataset(np.arange(n))
        self.check_create_empty_dataset(np.arange(n * 5 - 1))
