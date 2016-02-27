from chainer import cuda
from chainer import function
from chainer.utils import memory


class MemoryHook(function.FunctionHook):
    """Function hook that detects CUDA memory allocation and deallocation.

    Attributes:
        allocate_history: List of history of CUDA memory
            allocation and deallocation. It consists of
            3-tuples of the function that calls this function,
            string that represents the event,
            and allocated or deallocated memory size
    """

    name = 'MemoryHook'

    def __init__(self):
        self.allocate_history = []

    def preprocess(self, function, in_data, out_grad=None):
        def p(event, size):
            self.allocate_history.append((function, event, size))

        self.prev_allocator = cuda.cuda.get_allocator()
        prof = memory.MemoryProfiler(self.prev_allocator, p)
        cuda.cuda.set_allocator(prof.malloc)

    def postprocess(self, function, in_data, out_grad=None):
        cuda.cuda.set_allocator(self.prev_allocator)

    def total_allocate_size(self):
        return sum(s if e == 'alloc' else 0
                   for _, e, s in self.allocate_history)

    def total_deallocate_size(self):
        return sum(s if e == 'dealloc' else 0
                   for _, e, s in self.allocate_history)
