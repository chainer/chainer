import chainerx
from chainerx import _pybind_cuda

try:
    import cupy
    _cupy_available = True
except Exception:
    _cupy_available = False


_chainerx_allocator = None


def cupy_share_allocator(owner=chainerx._global_context):
    # Replace CuPy's allocator with ChainerX's if ChainerX is available with
    # the CUDA backend. This is needed in order to share the GPU memory
    # without having both modules using separate memory pools.

    # TODO(imanishi): Make sure this allocator works when the global
    # default context is changed by the user. It currently will not
    # since the allocator is only configured here once.
    try:
        owner.get_backend('cuda')
    except chainerx.BackendError:
        raise RuntimeError(
            'Cannot share allocator with CuPy without the CUDA backend.')

    if not _cupy_available:
        raise RuntimeError(
            'Cannot share allocator with CuPy since CuPy is not available.')

    param = _pybind_cuda.get_backend_ptr()
    malloc_func, free_func = _pybind_cuda.get_backend_malloc_free_ptrs()

    global _chainerx_allocator
    _chainerx_allocator = cupy.cuda.memory.ExternalAllocator(
        param, malloc_func, free_func, owner)

    cupy.cuda.set_allocator(_chainerx_allocator.malloc)
