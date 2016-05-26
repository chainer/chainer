import time

import numpy

from chainer import cuda
from chainer import function


class Timer(object):

    def __init__(self, xp):
        self.xp = xp
        self.running = False
        self.reset()

    def reset(self):
        self.total_time = 0.0
        self.last_increment = None
        self.count = 0

    def start(self):
        if self.running:
            return
        if self.xp == numpy:
            self._start = time.time()
        else:
            self._start = cuda.Event()
            self._start.record()
        self.running = True

    def stop(self):
        if not self.running:
            return 0.0

        if self.xp == numpy:
            self._stop = time.time()
            elapsed_time = self._stop - self._start
        else:
            self._stop = cuda.Event()
            self._stop.record()
            self._stop.synchronize()
            # get_elapsed_time returns result in milliseconds
            elapsed_time = cuda.cupy.cuda.get_elapsed_time(
                self._start, self._stop) / 1000

        self.running = False
        self.total_time += elapsed_time
        self.last_increment = elapsed_time
        self.count += 1

        return elapsed_time

    def mean(self):
        if self.count == 0:
            raise ValueError('count is 0')
        else:
            return self.total_time / self.count


class TimerHook(function.FunctionHook):
    """Function hook for measuring elapsed time of functions.

    Attributes:
        call_history: List of measurement results. It consists of pairs of
            the function that calls this hook and the elapsed time
            the function consumes.
        pass_through_time: Stores the elapsed time within "with" statement.

    """

    name = 'TimerHook'

    def __init__(self, xp=numpy):
        self.call_history = []
        self.xp = xp

    def __enter__(self, *args, **kwargs):
        self.pass_through_timer = Timer(self.xp)
        self.pass_through_timer.start()
        return super(TimerHook, self).__enter__(*args, **kwargs)

    def __exit__(self, *args, **kwargs):
        self.pass_through_time = self.pass_through_timer.stop()
        return super(TimerHook, self).__exit__(*args, **kwargs)

    def _preprocess(self, xp):
        self.timer = Timer(xp)
        self.timer.start()

    def forward_preprocess(self, function, in_data):
        xp = cuda.get_array_module(*in_data)
        self._preprocess(xp)

    def backward_preprocess(self, function, in_data, out_grad):
        xp = cuda.get_array_module(*(in_data + out_grad))
        self._preprocess(xp)

    def _postprocess(self, function):
        elapsed_time = self.timer.stop()
        self.call_history.append((function, elapsed_time))

    def forward_postprocess(self, function, in_data):
        xp = cuda.get_array_module(*in_data)
        self._postprocess(function)

    def backward_postprocess(self, function, in_data, out_grad):
        xp = cuda.get_array_module(*(in_data + out_grad))
        self._postprocess(function)

    def total_time(self):
        """Returns total elapsed time in seconds."""
        return sum(t for (_, t) in self.call_history)
