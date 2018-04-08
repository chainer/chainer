import io
import unittest

from chainer import datasets


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
