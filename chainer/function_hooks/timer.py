import time

import numpy

from chainer import cuda
from chainer import function


class CPUTimer(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.elapsed_times = []
        self.running = False

    def __enter__(self, *args, **kwargs):
        self.reset()
        self.start()
        return self

    def __exit__(self, *args, **kwargs):
        self.stop()
        return self

    def start(self):
        if self.running:
            return
        self.start.append(time.time())
        self.running = True

    def stop(self):
        if not self.running:
            return
        self.stop.append(time.time())
        self.running = False

    def total_time(self):
        self.elapsed_times = map(stop - start for start, stop
                                 in zip(self.start, self.stop))
        return sum(self.elapsed_times)
        
    def count(self):
        return len(self.stop)

    def mean(self):
        if self.count == 0:
            raise ValueError('Cannot calculate the mean elapsed time '
                             'because this timer has never measure elapsed times.')
        else:
            return self.total_time() / self.count()


class GPUTimer(object):

    def __init__(self, blocking_method='non_blocking'):
        if not (blocking_method == 'non_block' or
                blocking_method == 'block_first_time' or
                blocking_method == 'block_every_time'):
            raise ValueError(
                'Invalid blocking method:{}'.format(blocking_method))
        self.blocking_method = blocking_method
        self.reset()

    def __enter__(self, *args, **kwargs):
        self.reset()
        self.start()
        return self

    def __exit__(self, *args, **kwargs):
        self.stop()
        return self

    def reset(self):
        self.running = False
        self.start_events = []
        self.stop_events = []
        self.elapsed_times = None
        self._synchronized = False

    def start(self):
        if self.running:
            return

        if self._synchronized:
            raise RuntimeError('Thit timer is already synchronized. '
                               'Please reset the timer first.')

        start = cuda.Event()
        start.record()
        if ((self.blocking_method == 'block_first_time' and
             not self.start_events) or
            (self.blocking_method == 'block_every_time')):
            start.synchronize()
        self.start_events.append(start)
        self.running = True

    def stop(self):
        if not self.running:
            return

        stop = cuda.Event()
        stop.record()
        self.stop_events.append(stop)
        self.running = False

    def synchronize(self):
        if self.running:
            raise RuntimeError('Timer is running.')
        if self._synchronized:
            return

        if len(self.stop_events) > 0:
            stop_events[-1].synchronize()
        self._synchronized = True
        self.elapsed_times = map(cuda.cupy.cuda.get_elapsed_time(start, stop) / 1000
                                 for start, stop in zip(self.start_events, self.stop_events))

    def total_time(self):
        self.synchronize()
        return sum(self.elapsed_times)

    @property
    def synchronized(self):
        return self._synchronized

    def count(self):
        """Returns number of measurements that is already finish recording."""
        return len(self.stop_events)

    def mean(self):
        if self.count == 0:
            raise ValueError('Cannot calculate the mean elapsed time '
                             'because this timer has never measure elapsed times.')
        else:
            return self.total_time() / self.count()


class TimerHook(function.FunctionHook):
    """Function hook for measuring elapsed time of functions.

    Attributes:
        call_history: List of measurement results. It consists of pairs of
            the function that calls this hook and the elapsed time
            the function consumes.
        pass_through_time: Stores the elapsed time within "with" statement.

    """

    name = 'TimerHook'

    def _preprocess(self, xp):
        self.timer = Timer()
        self.timer.start()

    def forward_preprocess(self, function, in_data):
        xp = cuda.get_array_module(*in_data)
        self._preprocess(xp)

    def backward_preprocess(self, function, in_data, out_grad):
        xp = cuda.get_array_module(*(in_data + out_grad))
        self._preprocess(xp)

    def _postprocess(self, function):
        self.timer.stop()
        self.call_history.append((function, self.timer.total_time()))

    def forward_postprocess(self, function, in_data):
        xp = cuda.get_array_module(*in_data)
        self._postprocess(function)

    def backward_postprocess(self, function, in_data, out_grad):
        xp = cuda.get_array_module(*(in_data + out_grad))
        self._postprocess(function)

    def total_time(self):
        """Returns total elapsed time in seconds."""
        return sum(t for (_, t) in self.call_history)
