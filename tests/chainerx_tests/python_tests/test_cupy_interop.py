# These tests are integration tests with CuPy.
import numpy
import pytest

import chainerx
import chainerx.testing


try:
    import cupy
except Exception:
    cupy = None


_fromrawpointer = chainerx._core._fromrawpointer


@pytest.mark.cuda()
def test_cupy_to_chainerx_contiguous():
    dtype = numpy.float32
    a_cupy = cupy.arange(6, dtype=dtype).reshape((2, 3))
    a_chx = _fromrawpointer(
        a_cupy.data.mem.ptr,
        a_cupy.shape,
        a_cupy.dtype,
        a_cupy.strides,
        'cuda:0',
        0,
        a_cupy)

    assert a_chx.device.name == 'cuda:0'
    chainerx.testing.assert_array_equal_ex(a_chx, a_cupy.get())

    # Write to a_cupy
    a_cupy[0, 1] = 8
    chainerx.testing.assert_array_equal_ex(
        a_chx, numpy.array([[0, 8, 2], [3, 4, 5]], dtype))

    # Write to a_chx
    a_chx += 1
    chainerx.testing.assert_array_equal_ex(
        a_cupy.get(), numpy.array([[1, 9, 3], [4, 5, 6]], dtype))


@pytest.mark.cuda()
def test_cupy_to_chainerx_delete_cupy_first():
    dtype = numpy.float32
    a_cupy = cupy.arange(6, dtype=dtype).reshape((2, 3))
    a_chx = _fromrawpointer(
        a_cupy.data.mem.ptr,
        a_cupy.shape,
        a_cupy.dtype,
        a_cupy.strides,
        'cuda:0',
        0,
        a_cupy)

    del a_cupy

    a_chx += 1
    chainerx.testing.assert_array_equal_ex(
        a_chx, numpy.array([[1, 2, 3], [4, 5, 6]], dtype))


@pytest.mark.cuda()
def test_cupy_to_chainerx_delete_chainerx_first():
    dtype = numpy.float32
    a_cupy = cupy.arange(6, dtype=dtype).reshape((2, 3))
    a_chx = _fromrawpointer(
        a_cupy.data.mem.ptr,
        a_cupy.shape,
        a_cupy.dtype,
        a_cupy.strides,
        'cuda:0',
        0,
        a_cupy)

    del a_chx

    a_cupy += 1
    chainerx.testing.assert_array_equal_ex(
        a_cupy.get(), numpy.array([[1, 2, 3], [4, 5, 6]], dtype))


@pytest.mark.cuda()
def test_cupy_to_chainerx_from_invalid_pointer():
    dtype = numpy.float32
    a_numpy = numpy.arange(6, dtype=dtype).reshape((2, 3))
    with pytest.raises(chainerx.ChainerxError):
        _fromrawpointer(
            a_numpy.ctypes.data,
            a_numpy.shape,
            a_numpy.dtype,
            a_numpy.strides,
            'cuda:0',
            0,
            a_numpy)


@pytest.mark.cuda()
def test_cupy_to_chainerx_noncontiguous_with_offset():
    dtype = numpy.float32
    a_cupy = cupy.arange(12, dtype=dtype).reshape((2, 6))[::-1, ::2]
    offset = a_cupy.data.ptr - a_cupy.data.mem.ptr

    # test preconditions
    assert offset > 0
    assert not a_cupy.flags.c_contiguous

    a_chx = _fromrawpointer(
        a_cupy.data.mem.ptr,
        a_cupy.shape,
        a_cupy.dtype,
        a_cupy.strides,
        'cuda:0',
        offset,
        a_cupy)

    assert a_chx.strides == a_cupy.strides
    chainerx.testing.assert_array_equal_ex(
        a_chx, a_cupy.get(), strides_check=False)

    a_cupy[1, 1] = 53

    assert a_chx.strides == a_cupy.strides
    chainerx.testing.assert_array_equal_ex(
        a_chx, a_cupy.get(), strides_check=False)


@pytest.mark.cuda()
def test_cupy_to_chainerx_noncontiguous_without_offset():
    # This test includes access to address before the given pointer (because of
    # a negative stride).
    dtype = numpy.float32
    a_cupy = cupy.arange(12, dtype=dtype).reshape((2, 6))[::-1, ::2]

    # test preconditons
    assert a_cupy.data.mem.ptr < a_cupy.data.ptr
    assert not a_cupy.flags.c_contiguous

    a_chx = _fromrawpointer(
        a_cupy.data.ptr,
        a_cupy.shape,
        a_cupy.dtype,
        a_cupy.strides,
        'cuda:0',
        0,
        a_cupy)

    assert a_chx.strides == a_cupy.strides
    chainerx.testing.assert_array_equal_ex(
        a_chx, a_cupy.get(), strides_check=False)

    a_cupy[1, 1] = 53

    assert a_chx.strides == a_cupy.strides
    chainerx.testing.assert_array_equal_ex(
        a_chx, a_cupy.get(), strides_check=False)


@pytest.mark.cuda(2)
def test_cupy_to_chainerx_nondefault_device():
    dtype = numpy.float32
    with cupy.cuda.Device(1):
        a_cupy = cupy.arange(6, dtype=dtype).reshape((2, 3))
    a_chx = _fromrawpointer(
        a_cupy.data.mem.ptr,
        a_cupy.shape,
        a_cupy.dtype,
        a_cupy.strides,
        'cuda:1',
        0,
        a_cupy)

    assert a_chx.device.name == 'cuda:1'
    chainerx.testing.assert_array_equal_ex(a_chx, a_cupy.get())


@pytest.mark.cuda(2)
def test_cupy_to_chainerx_invalid_device():
    dtype = numpy.float32
    with cupy.cuda.Device(1):
        a_cupy = cupy.arange(6, dtype=dtype).reshape((2, 3))
    with pytest.raises(chainerx.ChainerxError):
        _fromrawpointer(
            a_cupy.data.mem.ptr,
            a_cupy.shape,
            a_cupy.dtype,
            a_cupy.strides,
            'cuda:0',
            0,
            a_cupy)


@pytest.mark.cuda()
def test_chainerx_to_cupy_contiguous():
    dtype = 'float32'
    a_chx = chainerx.arange(6, dtype=dtype, device='cuda:0').reshape((2, 3))
    a_cupy = cupy.ndarray(
        a_chx.shape,
        cupy.dtype(a_chx.dtype.name),
        cupy.cuda.MemoryPointer(cupy.cuda.UnownedMemory(
            a_chx.data_ptr + a_chx.offset,
            a_chx.data_size,
            a_chx,
            0), 0),
        strides=a_chx.strides,
    )

    assert a_cupy.device.id == 0
    chainerx.testing.assert_array_equal_ex(a_chx, a_cupy.get())

    # Write to a_cupy
    a_cupy[0, 1] = 8
    chainerx.testing.assert_array_equal_ex(
        a_chx, numpy.array([[0, 8, 2], [3, 4, 5]], dtype))

    # Write to a_chx
    a_chx += 1
    chainerx.testing.assert_array_equal_ex(
        a_cupy.get(), numpy.array([[1, 9, 3], [4, 5, 6]], dtype))


@pytest.mark.cuda()
def test_chainerx_to_cupy_delete_cupy_first():
    dtype = 'float32'
    a_chx = chainerx.arange(6, dtype=dtype, device='cuda:0').reshape((2, 3))
    a_cupy = cupy.ndarray(
        a_chx.shape,
        cupy.dtype(a_chx.dtype.name),
        cupy.cuda.MemoryPointer(cupy.cuda.UnownedMemory(
            a_chx.data_ptr + a_chx.offset,
            a_chx.data_size,
            a_chx,
            0), 0),
        strides=a_chx.strides,
    )

    del a_cupy

    a_chx += 1
    chainerx.testing.assert_array_equal_ex(
        a_chx, numpy.array([[1, 2, 3], [4, 5, 6]], dtype))


@pytest.mark.cuda()
def test_chainerx_to_cupy_delete_chainerx_first():
    dtype = 'float32'
    a_chx = chainerx.arange(6, dtype=dtype, device='cuda:0').reshape((2, 3))
    a_cupy = cupy.ndarray(
        a_chx.shape,
        cupy.dtype(a_chx.dtype.name),
        cupy.cuda.MemoryPointer(cupy.cuda.UnownedMemory(
            a_chx.data_ptr + a_chx.offset,
            a_chx.data_size,
            a_chx,
            0), 0),
        strides=a_chx.strides,
    )

    del a_chx

    a_cupy += 1
    chainerx.testing.assert_array_equal_ex(
        a_cupy.get(), numpy.array([[1, 2, 3], [4, 5, 6]], dtype))


@pytest.mark.cuda()
def test_chainerx_to_cupy_noncontiguous():
    dtype = 'float32'
    a_chx = chainerx.arange(
        12, dtype=dtype, device='cuda:0').reshape((2, 6))[::-1, ::2]
    offset = a_chx.offset

    # test preconditions
    assert offset > 0
    assert not a_chx.is_contiguous

    a_cupy = cupy.ndarray(
        a_chx.shape,
        cupy.dtype(a_chx.dtype.name),
        cupy.cuda.MemoryPointer(cupy.cuda.UnownedMemory(
            a_chx.data_ptr,
            a_chx.data_size,
            a_chx,
            0), offset),
        strides=a_chx.strides,
    )

    assert a_chx.strides == a_cupy.strides
    chainerx.testing.assert_array_equal_ex(
        a_chx, a_cupy.get(), strides_check=False)

    a_cupy[1, 1] = 53

    assert a_chx.strides == a_cupy.strides
    chainerx.testing.assert_array_equal_ex(
        a_chx, a_cupy.get(), strides_check=False)


@pytest.mark.cuda(2)
def test_chainerx_to_cupy_nondefault_device():
    dtype = 'float32'
    a_chx = chainerx.arange(6, dtype=dtype, device='cuda:1').reshape((2, 3))
    a_cupy = cupy.ndarray(
        a_chx.shape,
        cupy.dtype(a_chx.dtype.name),
        cupy.cuda.MemoryPointer(cupy.cuda.UnownedMemory(
            a_chx.data_ptr + a_chx.offset,
            a_chx.data_size,
            a_chx,
            -1), 0),
        strides=a_chx.strides,
    )

    assert a_cupy.device.id == 1
    chainerx.testing.assert_array_equal_ex(a_chx, a_cupy.get())
