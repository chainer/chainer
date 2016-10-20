import time

from chainer import cuda

import numpy


def get_timer(xp, *args, **kwargs):
    if xp is numpy:
        return CPUTimer(*args, **kwargs)
    elif xp is cuda.cupy:
        return GPUTimer(*args, **kwargs)
    else:
        raise ValueError('xp should be either NumPy or CuPy. '
                         'Actually it is {}'.format(xp))


class Timer(object):

    @property
    def xp(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def __enter__(self, *args, **kwargs):
        self.reset()
        self.start()
        return self

    def __exit__(self, *args, **kwargs):
        self.stop()
        return self

    def total_time(self):
        raise NotImplementedError

    def count(self):
        raise NotImplementedError

    def mean(self):
        count = self.count()
        if count == 0:
            raise ValueError('Cannot calculate the mean elapsed time '
                             'because this timer has never '
                             'measure elapsed times.')
        else:
            return self.total_time() / count


class CPUTimer(Timer):

    def __init__(self):
        self.reset()

    @property
    def xp(self):
        return numpy

    def reset(self):
        self.elapsed_times = []
        self.start_times = []
        self.stop_times = []
        self.running = False

    def start(self):
        if self.running:
            return
        self.start_times.append(time.time())
        self.running = True

    def stop(self):
        if not self.running:
            return
        self.stop_times.append(time.time())
        self.running = False

    def total_time(self):
        self.elapsed_times = list(stop - start for start, stop
                                  in zip(self.start_times, self.stop_times))
        return sum(self.elapsed_times)

    def count(self):
        return len(self.stop_times)


class GPUTimer(Timer):

    def __init__(self, blocking_method='non_block'):
        if not cuda.available:
            raise RuntimeError('CUDA must be available to use GPUTimer.')

        if not (blocking_method == 'non_block' or
                blocking_method == 'block_first_time' or
                blocking_method == 'block_every_time'):
            raise ValueError(
                'Invalid blocking method:{}'.format(blocking_method))
        self.blocking_method = blocking_method
        self.reset()

    @property
    def xp(self):
        return cuda.cupy

    def reset(self):
        self.running = False
        self.start_events = []
        self.stop_events = []
        self.elapsed_times = []
        self.synchronized = False

    def _need_synchronization_before_measurement(self):
        return ((self.blocking_method == 'block_first_time' and
                 not self.start_events) or
                (self.blocking_method == 'block_every_time'))

    def start(self):
        if self.running:
            return

        if self.synchronized:
            raise RuntimeError('This timer is already synchronized. '
                               'Please reset the timer first.')

        start = cuda.Event()
        stop = cuda.Event()

        start.record()

        if self._need_synchronization_before_measurement():
            start.synchronize()

        self.start_events.append(start)
        self.stop_events.append(stop)
        self.running = True

    def stop(self):
        if not self.running:
            return

        self.stop_events[-1].record()
        self.running = False

    def synchronize(self):
        if self.running:
            raise RuntimeError('Timer is running.')
        if self.synchronized:
            return

        if len(self.stop_events) > 0:
            self.stop_events[-1].synchronize()
        self.synchronized = True
        self.elapsed_times = list(
            cuda.cupy.cuda.get_elapsed_time(start, stop) / 1000
            for start, stop in zip(self.start_events, self.stop_events))

    def total_time(self):
        self.synchronize()
        return sum(self.elapsed_times)

    def count(self):
        """Returns number of measurements that already finish recording."""
        ret = len(self.stop_events)
        if self.running:
            ret -= 1
        return ret
