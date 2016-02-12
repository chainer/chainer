import os
import random
import six
import sys

import numpy

from chainer import cuda


dataset_root = os.environ.get(
    'CHAINER_DATASET_ROOT',
    os.path.join(os.environ['HOME'], '.chainer/datasets'))


def set_dataset_root(path):
    global dataset_root
    dataset_root = path


def get_dataset_path(name):
    return os.path.join(dataset_root, name)


class BatchIterator(object):

    """Default data iterator.

    TODO(beam2d): document it.

    """
    def __init__(self, dataset, batchsize=1, repeat=True, auto_shuffle=True,
                 device=None):
        self._dataset = dataset
        self._batchsize = batchsize
        self._repeat = repeat
        self._end_nonrepeat = False
        self.epoch = 0
        self.auto_shuffle = auto_shuffle
        self._device = device

        self._order = list(six.moves.range(len(dataset)))
        self._i = 0

        if auto_shuffle:
            self._shuffle()

        self._finalized = False

    def __del__(self):
        if not self._finalized:
            self.finalize()

    def __iter__(self):
        return self

    def next(self):
        if self._end_nonrepeat:
            raise StopIteration
        dataset, order, i = self._dataset, self._order, self._i
        N = len(dataset)
        batch = []
        for _ in range(self._batchsize):
            batch.append(dataset[order[i]])
            i += 1
            if i >= N:
                if not self._repeat:
                    self._end_nonrepeat = True
                    break
                self.epoch += 1
                if self.auto_shuffle:
                    self._shuffle()
                i = 0
        self._i = i
        return build_minibatch(batch, self._device)

    def finalize(self):
        self._finalized = True

    def serialize(self, serializer):
        self._end_nonrepeat = serializer('_end_nonrepeat', self._end_nonrepeat)
        self.epoch = serializer('epoch', self.epoch)
        self._order = list(serializer('_order', self._order))
        self._i = serializer('_i', self._i)

    def _shuffle(self):
        random.shuffle(self._order)


class Dataset(object):

    """Base class of all datasets.

    TODO(beam2d): document it.

    """
    @property
    def name(self):
        raise NotImplementedError

    def get_batch_iterator(self, batchsize=1, repeat=True, auto_shuffle=True,
                           device=None):
        return BatchIterator(self, batchsize, repeat, auto_shuffle, device)

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, i):
        raise NotImplementedError


def build_minibatch(examples, device=None):
    ret_tuple = isinstance(examples[0], tuple)
    if not ret_tuple:
        examples = [(example,) for example in examples]

    tuple_len = len(examples[0])
    if any([len(example) != tuple_len for example in examples]):
        raise ValueError('tuple length mismatched between batch elements')

    if device is None:
        xp = cuda.get_array_module(examples[0][0])
        def to_device(x):
            return x
    elif device < 0:
        xp = numpy
        to_device = cuda.to_cpu
    else:
        xp = cuda.cupy
        to_device = cuda.to_gpu

    cols = [[example[i] for example in examples]
            for i in six.moves.range(tuple_len)]  # transpose

    with cuda.get_device(device if xp is cuda.cupy else None):
        cols = [xp.concatenate([to_device(x)[None] for x in col])
                for col in cols]

    if ret_tuple:
        return tuple(cols)
    else:
        return cols[0]
