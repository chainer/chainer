import time

from chainer import cuda
from chainer import function


class TimerHook(function.FunctionHook):

    def preprocess(self, function, in_data):
        self.xp = cuda.get_array_module(*in_data)
        if self.xp == numpy:
            self.start = time.time()
        else:
            self.start = cuda.Event()
            self.stop = cuda.Event()
            self.start.record()

    def __call__(self, function, in_data):
        if self.xp == numpy:
            self.end = time.time()
            elapsed_time = self.end - self.start
        else:
            self.end.record()
            self.end.synchronize()
            elapsed_time = cuda.get_elapsed_time(self.start, self.end)

        print('{}\t{}'.format(function.label, elapsed_time))
