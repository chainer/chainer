import ctypes
import unittest

from cupy.cuda import memory
from cupy import testing


class MockMemory(memory.Memory):
    cur_ptr = 1

    def __init__(self, size):
        self.ptr = MockMemory.cur_ptr
        MockMemory.cur_ptr += size
        self.size = size
        self._device = None

    def __del__(self):
        self.ptr = 0
        pass


def mock_alloc(size):
    mem = MockMemory(size)
    return memory.MemoryPointer(mem, 0)


# -----------------------------------------------------------------------------
# Memory pointer

@testing.gpu
class TestMemoryPointer(unittest.TestCase):

    def test_int(self):
        pval = MockMemory.cur_ptr
        memptr = mock_alloc(1)
        self.assertEqual(pval, int(memptr))

    def test_add(self):
        pval = MockMemory.cur_ptr
        memptr = mock_alloc(8)

        memptr2 = memptr + 4
        self.assertIsInstance(memptr2, memory.MemoryPointer)
        self.assertEqual(pval + 4, int(memptr2))

        memptr3 = 4 + memptr
        self.assertIsInstance(memptr3, memory.MemoryPointer)
        self.assertEqual(pval + 4, int(memptr3))

        memptr += 4
        self.assertIsInstance(memptr, memory.MemoryPointer)
        self.assertEqual(pval + 4, int(memptr))

    def test_sub(self):
        pval = MockMemory.cur_ptr
        memptr = mock_alloc(8) + 4

        memptr2 = memptr - 4
        self.assertIsInstance(memptr2, memory.MemoryPointer)
        self.assertEqual(pval, int(memptr2))

        memptr -= 4
        self.assertIsInstance(memptr, memory.MemoryPointer)
        self.assertEqual(pval, int(memptr))

    def test_copy_to_and_from_host(self):
        a_gpu = memory.alloc(4)
        a_cpu = ctypes.c_int(100)
        a_gpu.copy_from(ctypes.cast(ctypes.byref(a_cpu), ctypes.c_void_p), 4)
        b_cpu = ctypes.c_int()
        a_gpu.copy_to_host(
            ctypes.cast(ctypes.byref(b_cpu), ctypes.c_void_p), 4)
        self.assertEqual(b_cpu.value, a_cpu.value)

    def test_copy_from_device(self):
        a_gpu = memory.alloc(4)
        a_cpu = ctypes.c_int(100)
        a_gpu.copy_from(ctypes.cast(ctypes.byref(a_cpu), ctypes.c_void_p), 4)

        b_gpu = memory.alloc(4)
        b_gpu.copy_from(a_gpu, 4)
        b_cpu = ctypes.c_int()
        b_gpu.copy_to_host(
            ctypes.cast(ctypes.byref(b_cpu), ctypes.c_void_p), 4)
        self.assertEqual(b_cpu.value, a_cpu.value)

    def test_memset(self):
        a_gpu = memory.alloc(4)
        a_gpu.memset(1, 4)
        a_cpu = ctypes.c_ubyte()
        for i in range(4):
            a_gpu.copy_to_host(
                ctypes.cast(ctypes.byref(a_cpu), ctypes.c_void_p), 1)
            self.assertEqual(a_cpu.value, 1)
            a_gpu += 1


# -----------------------------------------------------------------------------
# Memory pool


@testing.gpu
class TestSingleDeviceMemoryPool(unittest.TestCase):

    def setUp(self):
        self.pool = memory.SingleDeviceMemoryPool(allocator=mock_alloc)

    def test_alloc(self):
        p1 = self.pool.malloc(10)
        p2 = self.pool.malloc(10)
        p3 = self.pool.malloc(20)
        self.assertNotEqual(p1.ptr, p2.ptr)
        self.assertNotEqual(p1.ptr, p3.ptr)
        self.assertNotEqual(p2.ptr, p3.ptr)

    def test_free(self):
        p1 = self.pool.malloc(10)
        ptr1 = p1.ptr
        del p1
        p2 = self.pool.malloc(10)
        self.assertEqual(ptr1, p2.ptr)

    def test_free_different_size(self):
        p1 = self.pool.malloc(10)
        ptr1 = p1.ptr
        del p1
        p2 = self.pool.malloc(20)
        self.assertNotEqual(ptr1, p2.ptr)

    def test_free_all_free(self):
        p1 = self.pool.malloc(10)
        ptr1 = p1.ptr
        del p1
        self.pool.free_all_free()
        p2 = self.pool.malloc(10)
        self.assertNotEqual(ptr1, p2.ptr)
