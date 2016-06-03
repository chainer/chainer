import numpy

from chainer import cuda
from chainer import function as function_
from chainer.utils import timer


class TimerHook(function_.FunctionHook):
    """Function hook for measuring elapsed time of functions.

    Attributes:
        call_history: List of measurement results. It consists of pairs of
            the function that calls this hook and the elapsed time
            the function consumes.

    """

    name = 'TimerHook'

    def __init__(self):
        self.call_history = []

    def _preprocess(self):
        self.timer = timer.get_timer(self.xp)
        self.timer.start()

        # For backward compatibility
        if self.xp is numpy:
            self.start = self.timer.start_times[-1]
        else:
            self.start = self.timer.start_events[-1]
            self.stop = self.timer.stop_events[-1]

    def forward_preprocess(self, function, in_data):
        self.xp = cuda.get_array_module(*in_data)
        self._preprocess()

    def backward_preprocess(self, function, in_data, out_grad):
        self.xp = cuda.get_array_module(*(in_data + out_grad))
        self._preprocess()

    def _postprocess(self, function):
        self.timer.stop()
        self.call_history.append((function, self.timer.total_time()))

        # For backward compatibility
        if self.xp is numpy:
            self.stop = self.timer.stop_times[-1]

    def forward_postprocess(self, function, in_data):
        xp = cuda.get_array_module(*in_data)
        assert xp is self.xp
        self._postprocess(function)

    def backward_postprocess(self, function, in_data, out_grad):
        xp = cuda.get_array_module(*(in_data + out_grad))
        assert xp is self.xp
        self._postprocess(function)

    def total_time(self):
        """Returns total elapsed time in seconds."""
        return sum(t for (_, t) in self.call_history)
