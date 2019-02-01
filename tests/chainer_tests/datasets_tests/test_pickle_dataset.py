import io
import os
import sys
import unittest

from chainer import datasets
from chainer import testing
from chainer import utils


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
        self.tempdir = utils.tempdir()
        dirpath = self.tempdir.__enter__()
        self.path = os.path.join(dirpath, 'test.pkl')

    def tearDown(self):
        self.tempdir.__exit__(*sys.exc_info())

    def test_write_read(self):
        with datasets.open_pickle_dataset_writer(self.path) as writer:
            writer.write(1)

        with datasets.open_pickle_dataset(self.path) as dataset:
            assert dataset[0] == 1


testing.run_module(__name__, __file__)
