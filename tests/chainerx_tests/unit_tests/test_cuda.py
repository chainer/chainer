import pytest

from chainerx import _cuda


try:
    import cupy
except Exception:
    cupy = None


class CupyTestMemoryHook(cupy.cuda.memory_hook.MemoryHook):

    name = 'CupyTestMemoryHook'

    def __init__(self):
        self.used_bytes = 0
        self.acquired_bytes = 0

    def alloc_preprocess(self, **kwargs):
        self.acquired_bytes += kwargs['mem_size']

    def malloc_preprocess(self, **kwargs):
        self.used_bytes += kwargs['mem_size']


@pytest.mark.cuda()
def test_cupy_share_allocator():
    with CupyTestMemoryHook() as hook:
        cp_allocated = cupy.arange(10)

        used_bytes = hook.used_bytes
        acquired_bytes = hook.acquired_bytes
        assert used_bytes > 0
        assert acquired_bytes > 0

        # Create a new array after changing the allocator to the memory pool
        # of ChainerX and make sure that no additional memory has been
        # allocated by CuPy.

        _cuda.cupy_share_allocator()

        chx_allocated = cupy.arange(10)

        cupy.testing.assert_array_equal(cp_allocated, chx_allocated)

        assert used_bytes == hook.used_bytes
        assert acquired_bytes == hook.acquired_bytes
