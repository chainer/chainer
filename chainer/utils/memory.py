from chainer import cuda


class MemoryProfiler(object):

    def __init__(self, malloc, listener):
        assert callable(listener)
        self._malloc = malloc
        self._listener = listener

    def malloc(self, size):
        self._listener('alloc', size)
        ptr = self._malloc(size)
        pmem = ProfileMemory(ptr.mem, self._listener)
        return cuda.cuda.MemoryPointer(pmem, 0)


class ProfileMemory(cuda.cuda.MemoryBase):

    def __init__(self, memory, listener):
        super(ProfileMemory, self).__init__(
            memory.size, memory.device, memory.ptr)

        self._memory = memory
        self._listener = listener

    def __del__(self):
        self._listener('dealloc', self.size)
