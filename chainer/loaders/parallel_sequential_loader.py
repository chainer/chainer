from chainer import dataset


class ParallelSequentialLoader(dataset.Dataset):

    """Dataset adapter to load one long sequence in parallel.

    TODO(beam2d): document it.

    """
    def __init__(self, baseset, batchsize):
        self._baseset = baseset
        self._batchsize = batchsize
        self._gap = len(baseset) // batchsize

    def __len__(self):
        return len(self._baseset)

    def __getitem__(self, i):
        j = i // self._batchsize
        k = i % self._batchsize
        return self._baseset[k * self._gap + j]
