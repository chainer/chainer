import os
import random
import six

import numpy

from chainer import cuda


dataset_root = os.environ.get('CHAINER_DATASET_ROOT',
                              os.path.expanduser('~/.chainer/datasets'))


def set_dataset_root(path):
    """Sets the root path of Chainer datasets.

    Chainer uses the environment variable ``CHAINER_DATASET_ROOT`` (if given)
    or the default path ``~/.chainer/datasets`` as the root path of datasets.
    This function changes the path to save/load datasets.

    The actual path to save a specific dataset can be obtained by the
    :func:`get_dataset_path` function.

    Args:
        path (str): New root path into which the datasets are stored.

    """
    global dataset_root
    dataset_root = path


def get_dataset_path(name):
    """Returns the path to a specific dataset.

    The path is generated from the dataset root path and given name.

    Args:
        name (str): Name of the dataset.

    Returns:
        str: The path to the dataset.

    """
    return os.path.join(dataset_root, name)


class BatchIterator(object):

    """Default batch iterator.

    This is the default implementation of batch iterators for :class:`Dataset`.
    Batch iterators should provide the following operators and methods:

    - :meth:`__iter__`
    - :meth:`next`
    - :meth:`finalize`
    - :meth:`serialize`

    Args:
        dataset (Dataset): Dataset to iterate over.
        batchsize (int): Number of data points in each minibatch.
        repeat (bool): If True, then this iterator loops over the dataset
            inifinitely.
        auto_shuffle (bool): If True, then this iterator shuffles the order of
            iteration for each epoch. Otherwise, it visits the data points in
            the order of indexes.
        device: Device specifier. Minibatches are sent to this device. Negative
            values indicate CPU. If this is None, arrays are not copied across
            CPU/GPUs (i.e. each array given by the dataset is used as is).

    Attributes:
        epoch (int): The number of iterations done over the whole dataset.
        auto_shuffle (bool): If True, then this iterator shuffles the order of
            iteration for each epoch.

    """
    def __init__(self, dataset, batchsize=1, repeat=True, auto_shuffle=True,
                 device=None):
        if batchsize <= 0:
            raise ValueError('batchsize must be positive')
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

    def __iter__(self):
        """Returns self."""
        return self

    def next(self):
        """Returns the next minibatch.

        Returns:
            Array or tuple of arrays: The next minibatch.

        """
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
        """Finalizes the iterator object.

        The default implementation does nothing.

        """
        pass

    def serialize(self, serializer):
        """Serializes the iterator object.

        This method should (de)serialize information necessary to resume the
        iteration. Note that it should not (de)serialize information given at
        the initialization (e.g. the dataset object itself).

        """
        self._end_nonrepeat = serializer('_end_nonrepeat', self._end_nonrepeat)
        self.epoch = serializer('epoch', self.epoch)
        self._order = list(serializer('_order', self._order))
        self._i = serializer('_i', self._i)

    def _shuffle(self):
        random.shuffle(self._order)


def build_minibatch(examples, device=None):
    """Creates a minibatch from a sequence of data points.

    Args:
        examples: Sequence of data points. Each data point can be either an
            array or a tuple of arrays.
        device: Device specifier. The minibatch is copied to this device.
            Negative values indicate CPU. If this is None, then the minibatch
            is not copied across CPU/GPUs.

    Returns:
        Array if each data point is an array, otherwise a tuple of arrays.

    """
    if len(examples) == 0:
        raise ValueError('cannot create a minibatch from an empty sequence')
    ret_tuple = isinstance(examples[0], tuple)
    if not ret_tuple:
        examples = [(example,) for example in examples]

    tuple_len = len(examples[0])
    if any([len(example) != tuple_len for example in examples]):
        raise ValueError('tuple length mismatched between batch elements')

    if device is None:
        def to_device(x):
            return x

        xp = cuda.get_array_module(examples[0][0])
    elif device < 0:
        to_device = cuda.to_cpu
        xp = numpy
    else:
        to_device = cuda.to_gpu
        xp = cuda.cupy

    cols = [[example[i] for example in examples]
            for i in six.moves.range(tuple_len)]  # transpose

    with cuda.get_device(None if xp is numpy else device):
        cols = [xp.concatenate([to_device(x)[None] for x in col])
                for col in cols]

    if ret_tuple:
        return tuple(cols)
    else:
        return cols[0]


class Dataset(object):

    """Base class of all datasets.

    Dataset is a simple container of data points. Each data point must be
    either an array or a tuple of arrays (e.g. a feature vector and a label).
    If a data point is just an array, all the data points must have the same
    shape and dtype. If a data point consists of multiple arrays, these arrays
    can have different shapes and dtypes, while the shapes and dtypes must be
    same for all data points.

    Dataset provides two ways to access the data points.

    First is `direct addressing`, provided by the :meth:`__getitem__` and
    :meth:`__len__` operators. Every implementation must override these
    operators.

    Second is `batch iteration`. This is provided by batch iterators, which
    iterate over the dataset (sequentially or randomly) and extract a minibatch
    of data points at once. Batch iterators are provided by the
    :meth:`get_batch_iterator` method. The :class:`Dataset` class provides the
    default implementation of batch iterators using the direct addressing.

    There is a special type of datasets called `dataset adapters`. Dataset
    adapters extend existing datasets by providing customized accessing
    methods. Dataset adapters basically use the direct addressing to extract
    data points of the base dataset.

    Attributes:
        name: The name of the dataset. Implementation should provide this
            attribute.

    """
    name = 'anonymous dataset'

    def get_batch_iterator(self, batchsize=1, repeat=True, auto_shuffle=True,
                           device=None):
        """Creates a new batch iterator.

        Batch iterator is an iterable object that iterates over minibatches of
        data points. The access pattern can be configured by the arguments of
        this method.

        Args:
            batchsize (int): Size of each minibatch.
            repeat (bool): If True, then the iterator repeat iterations over
                the dataset infinitely. Otherwise, the iterator stops the
                iteration (i.e. raises :class:`StopIteration` exception) after
                all the data points are iterated. In this case, if
                ``batchsize`` does not divide the number of data points in the
                dataset, then the last batch have data points fewer than
                ``batchsize``.
            auto_shuffle (bool): If True, then the order of iteration is
                shuffled for every epoch (i.e. once for each iteration over the
                whole dataset). Otherwise, the iterator iterates the
                data points in the order of indexes.
            device: Device specifier. Each batch is sent to this device.
                Negative values indicate CPU. If this is None, arrays are not
                copied across CPU/GPUs.

        Returns:
            Batch iterator object. The default implementation returns a
            :class:`~chainer.dataset.BatchIterator` object. See
            :class:`~chainer.dataset.BatchIterator` for the interface that the
            batch iterator should provide.

        """
        return BatchIterator(self, batchsize, repeat, auto_shuffle, device)

    def __len__(self):
        """Returns the number of data points in the dataset.

        Implementations must override this operator.

        """
        raise NotImplementedError

    def __getitem__(self, i):
        """Extracts the ``i``-th data point.

        Implementations must override this operator.

        Args:
            i (int): Index of the data point.

        Returns:
            Array or tuple of arrays: The ``i``-th data point.

        """
        raise NotImplementedError
