import multiprocessing
import Queue
import random

import six

from chainer import dataset


class MultiprocessLoader(dataset.Dataset):

    """Dataset adapter to load a base dataset using multiple processes.

    This dataset adapter provides a parallelized version of batch iterators,
    which uses worker processes to access the base dataset in parallel.
    Note that the :meth:`~chainer.Dataset.__len__` nor
    :meth:`~chainer.Dataset.__getitem__` operators are not parallelized. Only
    the batch iterators are parallelized instead.

    The parallelization is done by the standard :mod:`multiprocessing` module.
    In order to access the base dataset by worker processes, the base dataset
    is sent to them in the standard way (i.e. pickling).

    The parallelized batch iterator has the same interface as the standard
    batch iterator, while it actually parallelizes the access to the base
    dataset using ``n_processes`` worker processes. On the call of the
    ``next()`` method, the iterator waits for the last batch, and before
    returning it, starts the extraction of the next batch. Therefore, the batch
    loading is effectively parallelized against subsequent procedures (e.g.
    training and evaluation).

    Args:
        baseset (Dataset): Base dataset. It must support pickling to safely
            send the content to worker processes.
        n_processes (int): Number of worker processes to be used by each batch
            iterator. If it is None, then it is set to the number of available
            processors.

    """
    def __init__(self, baseset, n_processes=None):
        if not isinstance(baseset, dataset.Dataset):
            raise TypeError('base dataset must be an instance of Dataset')
        self._base = baseset
        self._n_processes = n_processes

    @property
    def name(self):
        return self._base.name

    def get_batch_iterator(self, batchsize=1, repeat=True, auto_shuffle=True,
                           device=None):
        return MultiprocessBatchIterator(
            self, batchsize, repeat, auto_shuffle, device,
            self._n_processes)

    def __len__(self):
        return len(self._base)

    def __getitem__(self, i):
        return self._base[i]


class MultiprocessBatchIterator(object):

    def __init__(self, dataset, batchsize=1, repeat=True, auto_shuffle=True,
                 device=None, n_processes=None):
        self._dataset = dataset
        self._batchsize = batchsize
        self._repeat = repeat
        self._end_nonrepeat = False
        self.epoch = 0
        self.auto_shuffle = auto_shuffle
        self._device = device

        self._order = list(six.moves.range(len(dataset)))
        self._i = 0
        self._i_pushed = None  # initialized at the first iteration

        if auto_shuffle:
            self._shuffle()

        if n_processes is None:
            try:
                n_processes = multiprocessing.cpu_count()
            except NotImplementedError:
                n_processes = 2

        queue_size = max(n_processes, batchsize)
        self._index_queue = multiprocessing.Queue(queue_size)
        self._data_queue = multiprocessing.Queue(queue_size)

        args = dataset, self._index_queue, self._data_queue
        self._workers = []
        for _ in range(n_processes):
            worker = multiprocessing.Process(target=_worker, args=args)
            worker.daemon = True
            self._workers.append(worker)
            worker.start()

        self._start = False
        self._finalized = False

    def __del__(self):
        if not self._finalized:
            self.finalize()

    def __iter__(self):
        return self

    @property
    def n_processes(self):
        return len(self._workers)

    def next(self):
        if self._end_nonrepeat:
            raise StopIteration
        if not self._start:
            # put at the first iteration
            self._put()
            self._start = True
        batch = self._get()
        self._put()  # prepare for the next iteration
        return dataset.build_minibatch(batch, self._device)

    def finalize(self):
        workers = self._workers
        self._workers = []
        try:
            while True:
                self._data_queue.get_nowait()
        except Queue.Empty:
            pass
        for _ in workers:
            self._index_queue.put(-1)  # termination signal
        for worker in workers:
            worker.join()

        self._finalized = True

    def serialize(self, serializer):
        self._end_nonrepeat = serializer('_end_nonrepeat', self._end_nonrepeat)
        self.epoch = serializer('epoch', self.epoch)
        self._order = list(serializer('_order', self._order))
        self._i = serializer('_i', self._i)

    def _shuffle(self):
        random.shuffle(self._order)

    def _put(self):
        N = len(self._dataset)
        i = self._i_pushed
        if i is None:  # first iteration
            i = self._i
        for k in range(self._batchsize):
            index = self._order[i]
            self._index_queue.put(index)
            i += 1
            if i >= N:
                i = 0
                if not self._repeat:
                    break
                elif self.auto_shuffle:
                    self._shuffle()
        self._i_pushed = i

    def _get(self):
        N = len(self._dataset)
        i = self._i
        batch = []
        for k in range(self._batchsize):
            batch.append(self._data_queue.get())
            i += 1
            if i >= N:
                if not self._repeat:
                    self._end_nonrepeat = True
                    break
                else:
                    i = 0
                    self.epoch += 1
        self._i = i
        return batch


def _worker(base, in_queue, out_queue):
    while True:
        index = in_queue.get()
        if index < 0:
            break
        out_queue.put(base[index])
