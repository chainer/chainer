from chainer import cuda
from chainer import function
from chainer.utils import memory


class MemoryHook(function.FunctionHook):

    def __init__(self):
        self.allocate_history = []

    def preprocess(self, function, in_data, out_grad=None):
        def p(event, size):
            self.allocate_history.append((event, size, function.label))

        self.prev_allocator = cuda.cuda.get_allocator()
        prof = memory.MemoryProfiler(self.prev_allocator, p)
        cuda.cuda.set_allocator(prof.malloc)

    def postprocess(self, function, in_data, out_grad=None):
        cuda.cuda.set_allocator(self.prev_allocator)

    def total_allocate_size(self):
        return sum(s if e == 'alloc' else 0 for e, s, _ in self.allocate_history)

    def total_deallocate_size(self):
        return sum(s if e == 'dealloc' else 0 for e, s, _ in self.allocate_history)
