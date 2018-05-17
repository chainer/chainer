import unittest

import mock
import numpy

import chainer
from chainer.backends import cuda
from chainer import _backprop_utils
from chainer import testing
from chainer.testing import attr


def make_array(start, shape, dtype):
    size = numpy.product(shape, dtype='i')
    a = numpy.arange(start, start + size)
    a = a.reshape(shape)
    a = a.astype(dtype, copy=False)
    return a


@testing.parameterize(*testing.product({
    'y_shape': [(4,), (0,), (2, 3), ()],
    'x_shape': [(3,), (0,), (4, 1), ()],
}))
class TestFunctionNode(unittest.TestCase):

    def _get_method(self, prefix, gpu):
        suffix = 'gpu' if gpu else 'cpu'
        return getattr(self.f, prefix + '_' + suffix)

    def setUp(self):
        y_shape = self.y_shape
        x_shape = self.x_shape
        y1 = make_array(1, y_shape, numpy.float32)
        y2 = make_array(2, y_shape, numpy.float32)
        gx1 = chainer.Variable(
            make_array(1, x_shape, numpy.float32))
        gx2 = None
        gy1 = make_array(1, y_shape, numpy.float32)
        gy2 = make_array(1, y_shape, numpy.float32)

        f = chainer.FunctionNode()
        f.check_type_forward = mock.MagicMock()
        f.forward_cpu = mock.MagicMock(return_value=(y1, y2))
        f.forward_gpu = mock.MagicMock()
        f.backward = mock.MagicMock(return_value=(gx1, gx2))
        self.f = f

        self.x1 = make_array(0, x_shape, numpy.float32)
        self.x2 = make_array(0, x_shape, numpy.int32)
        self.y1 = y1
        self.y2 = y2
        self.gx1 = gx1
        self.gx2 = gx2
        self.gx1_orig = chainer.Variable(
            make_array(3, x_shape, numpy.float32))
        self.gx2_orig = chainer.Variable(
            make_array(2, x_shape, numpy.float32))
        self.gx1_accum = gx1 + self.gx1_orig
        self.gy1 = gy1
        self.gy2 = gy2

    def tearDown(self):
        # Set None to delete cuda array
        self.f = None
        self.y1 = None
        self.y2 = None
        self.gx1 = None

    def setup_gpu(self):
        self.x1 = cuda.to_gpu(self.x1)
        self.x2 = cuda.to_gpu(self.x2)
        self.y1 = cuda.to_gpu(self.y1)
        self.y2 = cuda.to_gpu(self.y2)
        self.gx1.to_gpu()
        self.gx1_orig.to_gpu()
        self.gx2_orig.to_gpu()
        self.gx1_accum.to_gpu()
        self.gy1 = cuda.to_gpu(self.gy1)
        self.gy2 = cuda.to_gpu(self.gy2)
        self.f.forward_gpu = mock.MagicMock(return_value=(self.y1, self.y2))
        self.f.backward = mock.MagicMock(return_value=(self.gx1, self.gx2))

    def check_backward(self, gxs):
        flag_none = gxs[0] is None
        self.f.backward_accumulate = chainer.FunctionNode.backward_accumulate

        x1 = chainer.Variable(self.x1)
        x2 = chainer.Variable(self.x2)
        self.f.inputs = (x1.node, x2.node)
        gxrefs = [[gx] if gx is not None else [] for gx in gxs]
        grad_outputs = (self.gy1, self.gy2)
        grad_inputs = dict(zip(self.f.inputs, gxrefs))
        _backprop_utils.backward(self.f, (0, 1), grad_outputs, grad_inputs)
        gx1 = _backprop_utils._reduce(gxrefs[0])
        gx2 = _backprop_utils._reduce(gxrefs[1])
        if flag_none:
            numpy.testing.assert_array_equal(cuda.to_cpu(gx1.data),
                                             cuda.to_cpu(self.gx1.data))
            self.assertIsNone(gx2)
        else:
            numpy.testing.assert_array_equal(cuda.to_cpu(gx1.data),
                                             cuda.to_cpu(self.gx1_accum.data))
            numpy.testing.assert_array_equal(cuda.to_cpu(gx2.data),
                                             cuda.to_cpu(self.gx2_orig.data))

    def test_backward_none_cpu(self):
        self.check_backward((None, None))

    @attr.gpu
    def test_backward_none_gpu(self):
        self.setup_gpu()
        self.check_backward((None, None))

    def test_backward_cpu(self):
        self.check_backward((self.gx1_orig, self.gx2_orig))

    @attr.gpu
    def test_backward_gpu(self):
        self.setup_gpu()
        self.check_backward((self.gx1_orig, self.gx2_orig))


testing.run_module(__name__, __file__)
