from chainer import cuda
from chainer import function
from chainer.utils import memory


class MemoryHook(function.FunctionHook):

    def f(self, message, function):
        def p(event, size):
            self.history.append((event, size, message, function.label))

        self.history = []
        self.prev_allocator = cuda.cuda.get_allocator()
        prof = memory.MemoryProfiler(self.prev_allocator, p)
        cuda.cuda.set_allocator(prof.malloc)
        return None

    def forward_preprocess(self, function, in_data, out_grad=None):
        return self.f('forward', function)

    def backward_preprocess(self, function, in_data, out_grad=None):
        return self.f('backward', function)

    def postprocess(self, function, in_data, out_grad=None):
        cuda.cuda.set_allocator(self.prev_allocator)
        return None
