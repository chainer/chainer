import unittest

import mock
import numpy
import six

import chainer
from chainer import _backprop_utils
from chainer.backends import cuda
from chainer import testing
from chainer.testing import attr


def make_array(start, shape, dtype):
    size = numpy.product(shape, dtype='i')
    a = numpy.arange(start, start + size)
    a = a.reshape(shape)
    a = a.astype(dtype, copy=False)
    return a


class FuncWithBackward(chainer.FunctionNode):

    def backward(self, target_input_indexes, grad_outputs):
        return self._mock_backward(target_input_indexes, grad_outputs)


class FuncWithBackwardAccumulate(chainer.FunctionNode):

    def backward_accumulate(self, target_input_indexes, grad_outputs,
                            grad_inputs):
        """Computes gradients w.r.t.\\  specified inputs and accumulates them.

        This method provides a way to fuse the backward computation and the
        gradient accumulations in the case that the multiple functions are
        applied to the same variable.

        Users have to override either of this method or :meth:`backward`.
        It is often simpler to implement :meth:`backward` and is recommended
        if you do not need to provide efficient gradient accumulation.

        Args:
            target_input_indexes (tuple of int): Indices of the input variables
                w.r.t. which the gradients are required. It is guaranteed that
                this tuple contains at least one element.
            grad_outputs (tuple of Variable): Gradients w.r.t. the output
                variables. If the gradient w.r.t. an output variable is not
                given, the corresponding element is ``None``.
            grad_inputs (tuple of Variable): Gradients w.r.t. the input
                variables specified by ``target_input_indexes``. These values
                are computed by other computation paths. If there is no
                gradient value existing for the variable, the corresponding
                element is ``None``. See also the note below.

        Returns:
            Tuple of variables that represent the gradients w.r.t. specified
            input variables. Unlike :meth:`backward`, the length of the tuple
            **must** be same as that of ``target_input_indices``.

        .. note::

           When the same variable is passed to the multiple input arguments of
           a function, only the first position of ``grad_inputs`` corresponding
           to these input arguments may contain the gradient variable
           corresponding to that input variable, and other entries are set to
           ``None``. This is an implementation-detail convention to avoid the
           complication of correctly accumulating gradients in such a case.
           This behavior might be changed in a future version.

        """
        assert isinstance(target_input_indexes, tuple)
        assert isinstance(grad_outputs, tuple)
        assert isinstance(grad_inputs, tuple)

        # The default implementation uses backward(). You can override this
        # method without using backward().
        gxs = self._mock_backward(target_input_indexes, grad_outputs)

        len_gxs = len(gxs)
        if len_gxs == len(self.inputs):
            gxs = tuple([gxs[i] for i in target_input_indexes])
        elif len_gxs != len(target_input_indexes):
            raise ValueError(
                'number of gradients returned by %s (%s) is incorrect.'
                % (self._impl_name, self.label))

        return tuple([gx if g_input is None else
                      g_input if gx is None else
                      gx + g_input
                      for gx, g_input in six.moves.zip(gxs, grad_inputs)])


@testing.parameterize(*testing.product({
    'y_shape': [(4,), (0,), (2, 3), ()],
    'x_shape': [(3,), (0,), (4, 1), ()],
    'override': ['backward', 'backward_accumulate'],
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

        f = {
            'backward': FuncWithBackward,
            'backward_accumulate': FuncWithBackwardAccumulate,
        }[self.override]()
        f._mock_backward = mock.MagicMock(return_value=(gx1, gx2))
        f.check_type_forward = mock.MagicMock()
        f.forward_cpu = mock.MagicMock(return_value=(y1, y2))
        f.forward_gpu = mock.MagicMock()
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
        self.f._mock_backward = mock.MagicMock(
            return_value=(self.gx1, self.gx2))

    def check_backprop_step(self, gxs):
        flag_none = gxs[0] is None

        x1 = chainer.Variable(self.x1)
        x2 = chainer.Variable(self.x2)
        self.f.inputs = (x1.node, x2.node)
        gxrefs = [[gx] if gx is not None else [] for gx in gxs]
        grad_outputs = (self.gy1, self.gy2)
        grad_inputs = dict(zip(self.f.inputs, gxrefs))
        _backprop_utils.backprop_step(
            self.f, (0, 1), grad_outputs, grad_inputs, True)
        if not chainer.configuration.config.lazy_grad_sum:
            # assert eager grad sum
            for gxref in gxrefs:
                self.assertLessEqual(len(gxref), 1)
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

    def test_backprop_step_none_cpu(self):
        self.check_backprop_step((None, None))

    @attr.gpu
    def test_backprop_step_none_gpu(self):
        self.setup_gpu()
        self.check_backprop_step((None, None))

    def test_backprop_step_cpu(self):
        self.check_backprop_step((self.gx1_orig, self.gx2_orig))

    @attr.gpu
    def test_backprop_step_gpu(self):
        self.setup_gpu()
        self.check_backprop_step((self.gx1_orig, self.gx2_orig))


testing.run_module(__name__, __file__)
