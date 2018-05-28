import six.moves.cPickle as pickle

from chainer.dataset import dataset_mixin


class PickleDatasetWriter(object):

    def __init__(self, io, protocol=pickle.HIGHEST_PROTOCOL):
        self.positions = []
        self.writer = io
        self.protocol = protocol

    def write(self, x):
        position = self.writer.tell()
        pickle.dump(x, self.writer, protocol=self.protocol)
        self.positions.append(position)

    def flush(self):
        self.writer.flush()


class PickleDataset(dataset_mixin.DatasetMixin):

    def __init__(self, reader):
        if not reader.seekable():
            raise ValueError('reader must support random access')

        self.reader = reader
        self.positions = []
        reader.seek(0)
        while True:
            position = reader.tell()
            try:
                pickle.load(reader)
            except EOFError:
                break
            self.positions.append(position)

    def __len__(self):
        return len(self.positions)

    def get_example(self, index):
        self.reader.seek(self.positions[index])
        return pickle.load(self.reader)
