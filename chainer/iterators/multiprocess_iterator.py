from __future__ import division
from collections import namedtuple
import multiprocessing
from multiprocessing import sharedctypes
import queue
import threading
import warnings

import numpy
import six

from chainer.dataset import iterator


_PrefetchState = namedtuple('_PrefetchState', (
    'current_position', 'epoch', 'is_new_epoch',
    'previous_epoch_detail', 'order'))


class MultiprocessIterator(iterator.Iterator):

    """Dataset iterator that loads examples in parallel.

    This is an implementation of :class:`~chainer.dataset.Iterator` that loads
    examples with worker processes. It uses the standard :mod:`multiprocessing`
    module to parallelize the loading. The dataset is sent to the worker
    processes in the standard way using pickle.

    Note that this iterator effectively prefetches the examples for the next
    batch asynchronously after the current batch is returned.

    This iterator saves ``-1`` instead of ``None`` in snapshots since some
    serializers do not support ``None``.

    Args:
        dataset (~chainer.dataset.Dataset): Dataset to iterate.
        batch_size (int): Number of examples within each batch.
        repeat (bool): If ``True``, it infinitely loops over the dataset.
            Otherwise, it stops iteration at the end of the first epoch.
        shuffle (bool): If ``True``, the order of examples is shuffled at the
            beginning of each epoch. Otherwise, examples are extracted in the
            order of indexes.
        n_processes (int): Number of worker processes. The number of CPUs is
            used by default.
        n_prefetch (int): Number of prefetch batches.
        shared_mem (int): The size of using shared memory per data.
            If ``None``, size is adjusted automatically.

    """

    def __init__(self, dataset, batch_size, repeat=True, shuffle=True,
                 n_processes=None, n_prefetch=1, shared_mem=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.repeat = repeat
        self.shuffle = shuffle

        self.n_processes = n_processes or multiprocessing.cpu_count()
        self.n_prefetch = max(n_prefetch, 1)
        self.shared_mem = shared_mem

        self._finalized = False

        self._comm = _Communicator(self.n_prefetch)
        self.reset()

        # Do not hold reference to this thread. Otherwise an unused iterator
        # never get GC'ed whenever the thread is alive, so you lose the
        # chance to stop the thread.
        prefetch_loop = _PrefetchLoop(
            self.dataset, self.batch_size, self.repeat, self.shuffle,
            self.n_processes, self.n_prefetch, self.shared_mem, self._comm)
        thread = threading.Thread(target=prefetch_loop, name='prefetch_loop')
        thread.start()

    def __next__(self):
        batch, prefetch_state = self._comm.get()
        (self.current_position, self.epoch, self.is_new_epoch,
            self._previous_epoch_detail, self._order) = prefetch_state
        if batch is None:
            raise StopIteration
        else:
            return batch

    next = __next__

    def __del__(self):
        if not self._finalized:
            self._finalized = True
            self._comm.terminate()

    finalize = __del__

    def __copy__(self):
        other = MultiprocessIterator(
            self.dataset, self.batch_size, self.repeat, self.shuffle,
            self.n_processes, self.n_prefetch, self.shared_mem)

        comm = self._comm
        del self._comm
        other.__dict__.update(self.__dict__)
        self._comm = comm
        other.__class__ = self.__class__

        other._set_prefetch_state()
        return other

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / len(self.dataset)

    @property
    def previous_epoch_detail(self):
        if self._previous_epoch_detail < 0:
            return None
        return self._previous_epoch_detail

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        try:
            serializer('order', self._order)
        except KeyError:
            serializer('_order', self._order)
        try:
            self._previous_epoch_detail = serializer(
                'previous_epoch_detail', self._previous_epoch_detail)
        except KeyError:
            # guess previous_epoch_detail for older version
            self._previous_epoch_detail = self.epoch + \
                (self.current_position - self.batch_size) / len(self.dataset)
            if self.epoch_detail > 0:
                self._previous_epoch_detail = max(
                    self._previous_epoch_detail, 0.)
            else:
                self._previous_epoch_detail = -1.
        self._set_prefetch_state()

    def reset(self):
        if self._finalized:
            raise NotImplementedError(
                'Reset of finalized MultiProcessIterator is currently not '
                'supported.')
        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False
        # use -1 instead of None internally.
        self._previous_epoch_detail = -1.
        if self.shuffle:
            self._order = numpy.random.permutation(len(self.dataset))
        else:
            self._order = None

        self._set_prefetch_state()

    def _set_prefetch_state(self):
        prefetch_state = _PrefetchState(
            current_position=self.current_position,
            epoch=self.epoch,
            is_new_epoch=self.is_new_epoch,
            previous_epoch_detail=self._previous_epoch_detail,
            order=self._order)
        self._comm.reset(prefetch_state)


class _Communicator(object):

    STATUS_CONTINUE = 0
    STATUS_RESET = 1
    STATUS_TERMINATE = 2

    def __init__(self, n_prefetch):
        self.n_prefetch = n_prefetch

        self._lock = threading.Lock()
        self._not_full_cond = threading.Condition(self._lock)
        self._batch_queue = queue.Queue()
        self._n_reserved = 0
        self._status = _Communicator.STATUS_CONTINUE

    # called from thread
    def get(self):
        batch, prefetch_state = self._batch_queue.get()
        with self._lock:
            self._n_reserved -= 1
            self._not_full_cond.notify()
        return batch, prefetch_state

    # called from iterator
    def reset(self, prefetch_state):
        with self._lock:
            self._status = _Communicator.STATUS_RESET
            self._prefetch_state = prefetch_state
            # ensure prefetcher won't stuck, and
            # remove all caches no more used
            n_reserved = self._n_reserved
            self._n_reserved = 0
            self._not_full_cond.notify()
        for _ in six.moves.range(n_reserved):
            self._batch_queue.get()

    # called from iterator
    def terminate(self):
        with self._lock:
            self._status = _Communicator.STATUS_TERMINATE
            # ensure prefetcher won't stuck, and
            # clean up queue
            n_reserved = self._n_reserved
            self._n_reserved = 0
            self._not_full_cond.notify()
        for _ in six.moves.range(n_reserved):
            self._batch_queue.get()

    # called from thread
    def reserve(self):
        with self._lock:
            if self._n_reserved == self.n_prefetch:
                self._not_full_cond.wait()

            status = self._status
            self._status = _Communicator.STATUS_CONTINUE

            if status != _Communicator.STATUS_TERMINATE:
                self._n_reserved += 1
            prefetch_state = None
            if status == _Communicator.STATUS_RESET:
                prefetch_state = self._prefetch_state

        return status, prefetch_state

    # called from thread
    def put(self, batch, prefetch_state):
        self._batch_queue.put((batch, prefetch_state))


class _PrefetchLoop(object):

    def __init__(self, dataset, batch_size, repeat, shuffle,
                 n_processes, n_prefetch, mem_size, comm):
        self.dataset = dataset
        self.batch_size = batch_size
        self.repeat = repeat
        self.shuffle = shuffle
        self.n_processes = n_processes
        self.mem_size = mem_size
        self.comm = comm

    def __call__(self):
        chunk_size = max(self.batch_size // self.n_processes // 2, 1)
        while True:
            self._allocate_shared_memory()
            pool = multiprocessing.Pool(
                processes=self.n_processes,
                initializer=_Fetch.setup,
                initargs=(self.dataset, self.mem_size, self.mem_bulk))
            try:
                mem_size = self.mem_size
                while self.mem_size == mem_size:

                    status, prefetch_state = self.comm.reserve()
                    if status == _Communicator.STATUS_RESET:
                        self.prefetch_state = prefetch_state
                    elif status == _Communicator.STATUS_TERMINATE:
                        return

                    indices = self._proceed()
                    if indices is None:  # stop iteration
                        batch = None
                    else:
                        data_it = pool.imap(
                            _Fetch.run, enumerate(indices), chunk_size)
                        batch, mem_size = self._extract_result(data_it)

                    self.comm.put(batch, self.prefetch_state)
            finally:
                pool.close()
                pool.join()
            self.mem_size = mem_size

    def _allocate_shared_memory(self):
        if self.mem_size:
            self.mem_bulk = \
                sharedctypes.RawArray('b', self.batch_size * self.mem_size)
        else:
            self.mem_bulk = None

    def _proceed(self):
        n = len(self.dataset)
        (pos, epoch, is_new_epoch,
            previous_epoch_detail, order) = self.prefetch_state

        if pos < self.batch_size and epoch > 0 and not self.repeat:
            return None  # stop iteration

        previous_epoch_detail = epoch + pos / n

        new_pos = pos + self.batch_size
        if new_pos < n:
            if order is None:
                indices = numpy.arange(pos, new_pos)
            else:
                indices = order[pos:new_pos]
            is_new_epoch = False
        else:
            new_pos = new_pos - n if self.repeat else 0

            if order is None:
                indices = numpy.arange(pos, n)
                if self.repeat:
                    indices = \
                        numpy.concatenate((indices, numpy.arange(new_pos)))
            else:
                indices = order[pos:n]
                if self.repeat:
                    order = numpy.random.permutation(n)
                    indices = \
                        numpy.concatenate((indices, order[:new_pos]))
            epoch += 1
            is_new_epoch = True

        self.prefetch_state = _PrefetchState(
            new_pos, epoch, is_new_epoch,
            previous_epoch_detail, order)
        return indices

    def _extract_result(self, data_it):
        measure_mode = self.mem_bulk is None

        batch = []
        max_size = 0
        for i, data in enumerate(data_it):
            if measure_mode:
                max_size = max(max_size, _measure(data))
                batch.append(data)
            else:
                batch.append(_unpack(data, self.mem_bulk))

        mem_size = max_size if measure_mode else self.mem_size
        return batch, mem_size


# To avoid restrictions with inherited states or pickled values.
# It doesn't mean thread unsafity; each process uses different address space.
class _Fetch(object):
    @classmethod
    def setup(cls, dataset, mem_size, mem_bulk):
        cls.dataset = dataset
        cls.mem_size = mem_size
        cls.mem_bulk = mem_bulk

    @classmethod
    def run(cls, inputs):
        i, index = inputs
        data = cls.dataset[index]
        if cls.mem_bulk is not None:
            offset = i * cls.mem_size
            limit = offset + cls.mem_size
            data = _pack(data, cls.mem_bulk, offset, limit)
        return data


class _PackedNdarray(object):

    def __init__(self, array, mem, offset):
        self.shape = array.shape
        self.dtype = array.dtype
        self.nbytes = array.nbytes
        self.size = array.size
        self.offset = offset
        total = self.offset + self.nbytes
        if total > len(mem):
            raise ValueError(
                'Shared memory size is too small. expect:{}, actual:{}'.format(
                    total, len(mem)))
        target = numpy.frombuffer(mem, self.dtype, self.size, self.offset)
        target[...] = array.ravel()

    def unpack(self, mem):
        ret = numpy.frombuffer(mem, self.dtype, self.size, self.offset)
        ret = ret.reshape(self.shape).copy()
        return ret


def _measure(data):
    expect = 0
    t = type(data)
    if t is tuple or t is list or t is dict:
        for v in data:
            if isinstance(v, numpy.ndarray):
                expect += v.nbytes
    return expect


def _pack(data, mem, offset, limit):
    if len(mem) == 0:
        return data
    t = type(data)
    over = False
    if t is tuple or t is list:
        ret = []
        for v in data:
            if isinstance(v, numpy.ndarray):
                if v.nbytes + offset > limit:
                    over = True
                else:
                    v = _PackedNdarray(v, mem, offset)
                    offset += v.nbytes
            ret.append(v)
        data = t(ret)
    elif t is dict:
        ret = {}
        for k, v in six.iteritems(data):
            if isinstance(v, numpy.ndarray):
                if v.nbytes + offset > limit:
                    over = True
                else:
                    v = _PackedNdarray(v, mem, offset)
                    offset += v.nbytes
            ret[k] = v
        data = ret
    if over:
        expect = _measure(data)
        warnings.warn(
            'Shared memory size is too small.\n' +
            'Please set shared_mem option for MultiprocessIterator.\n' +
            'Expect shared memory size: {} bytes.\n'.format(expect) +
            'Actual shared memory size: {} bytes.'.format(limit - offset),
            UserWarning)
    return data


def _unpack(data, mem):
    if len(mem) == 0:
        return data
    t = type(data)
    if t is tuple or t is list:
        ret = []
        for v in data:
            if isinstance(v, _PackedNdarray):
                v = v.unpack(mem)
            ret.append(v)
        data = t(ret)
    elif t is dict:
        ret = {}
        for k, v in six.iteritems(data):
            if isinstance(v, _PackedNdarray):
                v = v.unpack(mem)
            ret[k] = v
        data = ret
    return data
