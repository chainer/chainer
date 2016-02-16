import time

import numpy

from chainer import cuda
from chainer import function


class TimerHook(function.FunctionHook):

    def preprocess(self, function, in_data):
        xp = cuda.get_array_module(*in_data)
        if xp == numpy:
            self.start = time.time()
        else:
            self.start = cuda.Event()
            self.stop = cuda.Event()
            self.start.record()
        return None

    def postprocess(self, function, in_data):
        xp = cuda.get_array_module(*in_data)
        if xp == numpy:
            self.stop = time.time()
            elapsed_time = self.stop - self.start
        else:
            self.stop.record()
            self.stop.synchronize()
            elapsed_time = cuda.cupy.cuda.get_elapsed_time(
                self.start, self.stop)
        return elapsed_time

    def total_time(self):
        return sum(t if p == 'postprocess' else 0.0 for (_, p, _, t) in self.hook_history)
