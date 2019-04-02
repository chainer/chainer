from __future__ import division
import datetime
import multiprocessing
from multiprocessing import sharedctypes  # type: ignore
import signal
import sys
import threading
import warnings

import numpy
import six

from chainer.dataset import iterator
from chainer.iterators import _statemachine
from chainer.iterators.order_samplers import ShuffleOrderSampler


_response_time = 0.1


def _raise_timeout_warning():
    warnings.warn(
        'Stalled dataset is detected. '
        'See the documentation of MultiprocessIterator for common causes and '
        'workarounds:\n'
        'https://docs.chainer.org/en/stable/reference/generated/'
        'chainer.iterators.MultiprocessIterator.html',
        MultiprocessIterator.TimeoutWarning)


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

    .. note::

            When you are using OpenCV somewhere in your code and the
            ``MultiprocessIterator`` is used in the training code, the
            training loop may get stuck at some point. In such situation,
            there are several workarounds to prevent the process got stuck.

            1. Set the environment variable as follows: ``OMP_NUM_THREADS=1``
            2. Add ``cv2.setNumThreads(0)`` right after ``import cv2`` in your
               training script.
            3. Use :class:`~chainer.iterators.MultithreadIterator` instead of
               ``MultiprocessIterator``.

    Args:
        dataset (~chainer.dataset.Dataset): Dataset to iterate.
        batch_size (int): Number of examples within each batch.
        repeat (bool): If ``True``, it infinitely loops over the dataset.
            Otherwise, it stops iteration at the end of the first epoch.
        shuffle (bool): If ``True``, the order of examples is shuffled at the
            beginning of each epoch. Otherwise, examples are extracted in the
            order of indexes. If ``None`` and no ``order_sampler`` is given,
            the behavior is the same as the case with ``shuffle=True``.
        n_processes (int): Number of worker processes. The number of CPUs is
            used by default.
        n_prefetch (int): Number of prefetch batches.
        shared_mem (int): The size of using shared memory per data.
            If ``None``, size is adjusted automatically.
        dataset_timeout (float): :class:`MultiprocessIterator.TimeoutWarning`
            will be issued after this time in seconds elapsed in each dataset
            realization. ``None`` to disable the warning. You can turn this
            warning into an error by using :func:`warnings.simplefilter`::

                warnings.simplefilter(
                    'error',
                    chainer.iterators.MultiprocessIterator.TimeoutWarning)

        order_sampler (callable): A callable that generates the order
            of the indices to sample in the next epoch when a epoch finishes.
            This function should take two arguments: the current order
            and the current position of the iterator.
            This should return the next order. The size of the order
            should remain constant.
            This option cannot be used when ``shuffle`` is not ``None``.
        maxtasksperchild (int): Number of tasks a worker of prefetch process
            can complete before it will exit and be replaced with a fresh
            worker process, to enable unused resources to be freed. If
            ``None``, worker processes will live as long as the pool.

    """

    class TimeoutWarning(RuntimeWarning):
        pass

    _interruption_testing = False  # for testing
    _finalized = False
    _prefetch_loop = None
    _comm = None

    def __init__(self, dataset, batch_size, repeat=True, shuffle=None,
                 n_processes=None, n_prefetch=1, shared_mem=None,
                 order_sampler=None, dataset_timeout=30.0,
                 maxtasksperchild=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.repeat = repeat
        self.shuffle = shuffle
        self.n_processes = n_processes or multiprocessing.cpu_count()
        self.n_prefetch = max(n_prefetch, 1)
        self.shared_mem = shared_mem
        self.dataset_timeout = dataset_timeout
        self._maxtasksperchild = maxtasksperchild

        if self.shuffle is not None:
            if order_sampler is not None:
                raise ValueError('`shuffle` is not `None` and a custom '
                                 '`order_sampler` is set. Please set '
                                 '`shuffle` to `None` to use the custom '
                                 'order sampler.')
            else:
                if self.shuffle:
                    order_sampler = ShuffleOrderSampler()
        else:
            if order_sampler is None:
                order_sampler = ShuffleOrderSampler()
        self.order_sampler = order_sampler

        self._comm = _Communicator(self.n_prefetch, dataset_timeout)
        self.reset()

        self._prefetch_loop = _PrefetchLoop(
            self.dataset, self.batch_size, self.repeat,
            self.n_processes, self.n_prefetch, self.shared_mem,
            self._comm, self.order_sampler,
            self._interruption_testing, self._maxtasksperchild)
        # defer launching prefetch thread until creating the worker pool,
        # not to leave a background thread in forked processes.

    def __next__(self):
        measure_mode = False
        if self._prefetch_loop.thread is None:
            if self._prefetch_loop.measure_required():
                measure_mode = True
                batch, state = self._prefetch_loop.measure(
                    self.dataset_timeout)
            self._prefetch_loop.launch_thread()

        if not measure_mode:
            batch, state = self._comm.get()

        self._previous_epoch_detail = self.epoch_detail
        self._state = state

        if batch is None:
            raise StopIteration
        else:
            return batch

    next = __next__

    def finalize(self):
        if self._finalized:
            return

        if self._comm is not None:
            self._comm.terminate()

        if self._prefetch_loop is not None:
            self._prefetch_loop.terminate()

        self._comm = None
        self._prefetch_loop = None
        self._finalized = True

    def __copy__(self):
        # This function is implemented for backward compatibility.
        # Please use `reset` normally.
        other = MultiprocessIterator(
            self.dataset, self.batch_size, self.repeat, shuffle=None,
            n_processes=self.n_processes, n_prefetch=self.n_prefetch,
            shared_mem=self.shared_mem, order_sampler=self.order_sampler)

        other._reset_state(self.current_position, self.epoch,
                           self.is_new_epoch, self._state.order)
        other._previous_epoch_detail = self._previous_epoch_detail
        return other

    @property
    def current_position(self):
        return self._state.current_position

    @property
    def epoch(self):
        return self._state.epoch

    @property
    def is_new_epoch(self):
        return self._state.is_new_epoch

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / self._epoch_size

    @property
    def previous_epoch_detail(self):
        if self._previous_epoch_detail < 0:
            return None
        return self._previous_epoch_detail

    def serialize(self, serializer):
        current_position = serializer('current_position',
                                      self.current_position)
        epoch = serializer('epoch', self.epoch)
        is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        order = self._state.order.copy()
        try:
            serializer('order', order)
        except KeyError:
            serializer('_order', order)
        self._reset_state(current_position, epoch, is_new_epoch, order)
        try:
            self._previous_epoch_detail = serializer(
                'previous_epoch_detail', self._previous_epoch_detail)
        except KeyError:
            # guess previous_epoch_detail for older version
            self._previous_epoch_detail = self.epoch + \
                (self.current_position - self.batch_size) / self._epoch_size
            if self.epoch_detail > 0:
                self._previous_epoch_detail = max(
                    self._previous_epoch_detail, 0.)
            else:
                self._previous_epoch_detail = -1.

    def reset(self):
        if self.order_sampler is None:
            order = None
        else:
            order = self.order_sampler(numpy.arange(len(self.dataset)), 0)
        self._reset_state(0, 0, False, order)
        self._previous_epoch_detail = -1.

    def _reset_state(self, current_position, epoch, is_new_epoch, order):
        if self._finalized:
            raise NotImplementedError(
                'Reset of finalized MultiProcessIterator is currently not '
                'supported.')
        self._state = _statemachine.IteratorState(
            current_position, epoch, is_new_epoch, order)
        self._comm.reset(self._state)

    @property
    def _epoch_size(self):
        order = self._state.order
        if order is None:
            epoch_size = len(self.dataset)
        else:
            epoch_size = len(order)
        return epoch_size


class _Communicator(object):

    STATUS_CONTINUE = 0
    STATUS_RESET = 1
    STATUS_TERMINATE = 2

    def __init__(self, n_prefetch, dataset_timeout):
        self.n_prefetch = n_prefetch
        self.dataset_timeout = dataset_timeout

        self._lock = threading.Lock()
        self._not_empty_cond = threading.Condition(self._lock)
        self._not_full_cond = threading.Condition(self._lock)
        self._batch_queue = []
        self._status = _Communicator.STATUS_CONTINUE
        self._reset_count = 0

    @property
    def is_terminated(self):
        with self._lock:
            return self._status == _Communicator.STATUS_TERMINATE

    # called from iterator
    def get(self):
        with self._lock:
            start = datetime.datetime.now()
            while len(self._batch_queue) == 0:
                self._not_empty_cond.wait(_response_time)
                dt = datetime.datetime.now() - start
                if (self.dataset_timeout is not None
                        and dt > datetime.timedelta(
                            seconds=self.dataset_timeout)):
                    _raise_timeout_warning()
            batch, prefetch_state = self._batch_queue.pop(0)
            self._not_full_cond.notify()
            return batch, prefetch_state

    # called from iterator
    def reset(self, prefetch_state):
        with self._lock:
            self._status = _Communicator.STATUS_RESET
            self._prefetch_state = prefetch_state
            self._batch_queue = []
            self._not_full_cond.notify()
            self._reset_count += 1

    # called from iterator
    def terminate(self):
        with self._lock:
            self._status = _Communicator.STATUS_TERMINATE
            self._batch_queue = []
            self._not_full_cond.notify()
            self._reset_count += 1

    # called from thread
    def check(self):
        with self._lock:
            status = self._status
            self._status = _Communicator.STATUS_CONTINUE
            prefetch_state = None
            if status == _Communicator.STATUS_RESET:
                prefetch_state = self._prefetch_state
            return status, prefetch_state, self._reset_count

    # called from thread
    def put(self, batch, prefetch_state, reset_count):
        with self._lock:
            if len(self._batch_queue) == self.n_prefetch:
                self._not_full_cond.wait()
            if reset_count == self._reset_count:
                self._batch_queue.append((batch, prefetch_state))
                self._not_empty_cond.notify()


class _PrefetchLoop(object):

    _thread = None
    _pool = None
    _terminating = False

    def __init__(self, dataset, batch_size, repeat,
                 n_processes, n_prefetch, mem_size, comm,
                 order_sampler,
                 _interruption_testing, maxtasksperchild):
        self.dataset = dataset
        self.batch_size = batch_size
        self.repeat = repeat
        self.n_processes = n_processes
        self.mem_size = mem_size
        self._comm = comm
        self.order_sampler = order_sampler
        self.maxtasksperchild = maxtasksperchild

        self._allocate_shared_memory()

        self._interruption_testing = _interruption_testing

    def terminate(self):
        self._terminating = True

        # Terminate the thread first because it depends on the pool.
        if self._thread is not None:
            while self._thread.is_alive():
                self._thread.join(_response_time)

        if self._pool is not None:
            self._pool.terminate()

        self._thread = None
        self._pool = None

    @property
    def thread(self):
        return self._thread

    def measure_required(self):
        return self.mem_size is None

    def measure(self, dataset_timeout):
        # dataset_timeout: timeout in seconds or None

        status, prefetch_state, _ = self._comm.check()
        if status == _Communicator.STATUS_RESET:
            self.prefetch_state = prefetch_state

        self.prefetch_state, indices = _statemachine.iterator_statemachine(
            self.prefetch_state, self.batch_size, self.repeat,
            self.order_sampler, len(self.dataset))
        if indices is None:  # stop iteration
            batch = None
        else:
            batch_ret = [None]

            def fetch_batch():
                batch_ret[0] = [self.dataset[idx] for idx in indices]

            if dataset_timeout is None:
                # Timeout is not set: fetch synchronously
                fetch_batch()
            else:
                # Timeout is set: fetch asynchronously and watch for timeout
                thr = threading.Thread(target=fetch_batch)
                thr.daemon = True
                thr.start()
                thr.join(dataset_timeout)
                if thr.is_alive():
                    _raise_timeout_warning()
                thr.join()

            batch = batch_ret[0]
            self.mem_size = max(map(_measure, batch))
            self._allocate_shared_memory()

        return batch, self.prefetch_state

    def _allocate_shared_memory(self):
        if self.measure_required():
            self.mem_bulk = None
        else:
            self.mem_bulk = \
                sharedctypes.RawArray('b', self.batch_size * self.mem_size)

    def launch_thread(self):
        self._pool = multiprocessing.Pool(
            processes=self.n_processes,
            initializer=_fetch_setup,
            initargs=(self.dataset, self.mem_size, self.mem_bulk),
            maxtasksperchild=self.maxtasksperchild)
        if self._interruption_testing:
            pids = self._pool.map(_report_pid, range(self.n_processes))
            print(' '.join(map(str, pids)))
            sys.stdout.flush()

        thread = threading.Thread(target=self._run, name='prefetch_loop')
        thread.setDaemon(True)
        thread.start()
        self._thread = thread
        return thread

    def _run(self):
        # The entry routine of the prefetch thread.

        alive = True
        try:
            while alive:
                if self._terminating:
                    break
                alive = self._task()
        finally:
            self._pool.close()
            self._pool.join()

    def _task(self):
        # Do a single task in the prefetch thread.
        # Returns a bool indicating whether the loop should continue running.

        status, prefetch_state, reset_count = self._comm.check()
        if status == _Communicator.STATUS_RESET:
            self.prefetch_state = prefetch_state
        elif status == _Communicator.STATUS_TERMINATE:
            return False  # stop loop

        self.prefetch_state, indices = _statemachine.iterator_statemachine(
            self.prefetch_state, self.batch_size, self.repeat,
            self.order_sampler, len(self.dataset))
        if indices is None:  # stop iteration
            batch = None
        else:
            future = self._pool.map_async(_fetch_run, enumerate(indices))
            while True:
                try:
                    data_all = future.get(_response_time)
                except multiprocessing.TimeoutError:
                    if self._comm.is_terminated:
                        return False
                else:
                    break
            batch = [_unpack(data, self.mem_bulk) for data in data_all]

        self._comm.put(batch, self.prefetch_state, reset_count)
        return True


# Using `parameterized` function (e.g. bound method) with Pool is tricky due to
# restrictions imposed by Pickle. Picklable types differ across versions.
# Just using top-level function with globals seems to be safest.
# it doesn't mean thread safety broken or global variables visible;
# notice that each process uses different address space.
# To make static linter happy, we first initialize global variables.
_fetch_dataset = None
_fetch_mem_size = None
_fetch_mem_bulk = None


def _fetch_setup(dataset, mem_size, mem_bulk):
    global _fetch_dataset, _fetch_mem_size, _fetch_mem_bulk
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _fetch_dataset = dataset
    _fetch_mem_size = mem_size
    _fetch_mem_bulk = mem_bulk


def _fetch_run(inputs):
    i, index = inputs
    data = _fetch_dataset[index]
    if _fetch_mem_bulk is not None:
        offset = i * _fetch_mem_size
        limit = offset + _fetch_mem_size
        data = _pack(data, _fetch_mem_bulk, offset, limit)
    return data


def _report_pid(_):  # for testing
    return multiprocessing.current_process().pid


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
    elif t is numpy.ndarray:
        if data.nbytes + offset > limit:
            over = True
        else:
            data = _PackedNdarray(data, mem, offset)
            offset += data.nbytes
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
    elif t is _PackedNdarray:
        data = data.unpack(mem)
    return data
