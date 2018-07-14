import io
import os
import tempfile
import unittest

from chainer import datasets
from chainer import testing


class TestPickleDataset(unittest.TestCase):

    def setUp(self):
        self.io = io.BytesIO()

    def test_write_read(self):
        writer = datasets.PickleDatasetWriter(self.io)
        writer.write(1)
        writer.write('hello')
        writer.write(1.5)
        writer.flush()

        dataset = datasets.PickleDataset(self.io)
        assert len(dataset) == 3
        assert dataset[0] == 1
        assert dataset[2] == 1.5
        assert dataset[1] == 'hello'


class TestPickleDatasetHelper(unittest.TestCase):

    def setUp(self):
        _, self.path = tempfile.mkstemp()

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_write_read(self):
        with datasets.open_pickle_dataset_writer(self.path) as writer:
            writer.write(1)

        with datasets.open_pickle_dataset(self.path) as dataset:
            assert dataset[0] == 1


testing.run_module(__name__, __file__)
