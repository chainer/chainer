import contextlib

from chainer import cuda


@contextlib.contextmanager
def memory_profile(info):
    prev = cuda.cuda.get_allocator()
    prof = MemoryProfiler(prev, info)
    cuda.cuda.set_allocator(prof.malloc)
    yield
    cuda.cuda.set_allocator(prev)


class MemoryProfiler(object):

    def __init__(self, malloc, info):
        self._info = info
        self._malloc = malloc

    def malloc(self, size):
        print('alloc', self._info, size)
        ptr = self._malloc(size)
        pmem = ProfileMemory(ptr.mem, self._info)
        return cuda.cuda.MemoryPointer(pmem, 0)


class ProfileMemory(cuda.cuda.MemoryBase):

    def __init__(self, memory, info):
        super(ProfileMemory, self).__init__(
            memory.size, memory.device, memory.ptr)

        self._memory = memory
        self._info = info

    def __del__(self):
        print('del', self._info, self.size)
