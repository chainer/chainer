class TestCopyTo(unittest.TestCase):

    def test_cpu_to_cpu(self):
        src = numpy.arange(1, 5, dtype=numpy.float32)
        dst = numpy.zeros_like(src)
        cuda.copyto(dst, src)
        numpy.testing.assert_array_equal(dst, src)

    @attr.gpu
    def test_cpu_to_gpu(self):
        src = numpy.arange(1, 5, dtype=numpy.float32)
        dst = cuda.cupy.zeros_like(src)
        cuda.copyto(dst, src)
        cuda.cupy.testing.assert_array_equal(dst, src)

    @attr.gpu
    def test_gpu_to_cpu(self):
        src = cuda.cupy.arange(1, 5, dtype=numpy.float32)
        dst = numpy.zeros_like(src.get())
        cuda.copyto(dst, src)
        cuda.cupy.testing.assert_array_equal(dst, src)

    @attr.gpu
    def test_gpu_to_gpu(self):
        src = cuda.cupy.arange(1, 5, dtype=numpy.float32)
        dst = cuda.cupy.zeros_like(src)
        cuda.copyto(dst, src)
        cuda.cupy.testing.assert_array_equal(dst, src)

    @attr.multi_gpu(2)
    def test_gpu_to_another_gpu(self):
        src = cuda.cupy.arange(1, 5, dtype=numpy.float32)
        with cuda.get_device_from_id(1):
            dst = cuda.cupy.zeros_like(src)
        cuda.copyto(dst, src)
        cuda.cupy.testing.assert_array_equal(dst, src)

    def test_fail_on_invalid_src(self):
        src = None
        dst = numpy.zeros(1)
        with self.assertRaises(TypeError):
            cuda.copyto(dst, src)

    def test_fail_on_invalid_dst(self):
        src = numpy.zeros(1)
        dst = None
        with self.assertRaises(TypeError):
            cuda.copyto(dst, src)
