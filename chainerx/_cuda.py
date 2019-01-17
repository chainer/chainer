import chainerx

try:
    # _pybind_cuda is unavailable if ChainerX is built without CUDA.
    from chainerx import _pybind_cuda
    _available = True
except Exception:
    _available = False

try:
    import cupy
    _cupy_available = True
except Exception:
    _cupy_available = False


def cupy_share_allocator():
    # Replace CuPy's allocator with ChainerX's if ChainerX is available with
    # the CUDA backend. This is needed in order to share the GPU memory
    # without having both modules using separate memory pools.

    if not _available:
        raise RuntimeError(
            'Cannot share allocator with CuPy without the CUDA backend.')
    if not _cupy_available:
        raise RuntimeError(
            'Cannot share allocator with CuPy since CuPy is not available.')

    c_allocator = _pybind_cuda.get_c_allocator()

    chainerx_allocator = cupy.cuda.memory.CFunctionAllocator(
        *c_allocator, chainerx._global_context)

    cupy.cuda.set_allocator(chainerx_allocator.malloc)
