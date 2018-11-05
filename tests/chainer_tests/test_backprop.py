import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import testing
import chainer.testing.backend
import chainerx


@testing.parameterize(*testing.product({
    'function_node': [True, False],
}))
@testing.backend.inject_backend_tests(
    None,
    [
        # CPU
        {},
        # CUDA
        {'use_cuda': True, 'cuda_device': 0},
        {'use_cuda': True, 'cuda_device': 1},
        # ChainerX
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
    ])
class TestFunctionBackprop(unittest.TestCase):

    def call_func_function(self, backend_config, x1, x2, x3):
        xp = backend_config.xp

        if xp is chainerx:
            backend_name, device_index = (
                backend_config.chainerx_device.split(':'))
            if backend_name == 'native':
                forward_xp = numpy
                backward_xp = numpy
            elif backend_name == 'cuda':
                forward_xp = cuda.cupy
                backward_xp = cuda.cupy
            else:
                assert False
        else:
            forward_xp = xp
            backward_xp = xp

        class Func(chainer.Function):
            def forward(self, inputs):
                # x1, x3: float32
                # x2: int32
                x1, x2, x3 = inputs

                assert isinstance(x1, forward_xp.ndarray)
                assert isinstance(x2, forward_xp.ndarray)
                assert isinstance(x3, forward_xp.ndarray)

                y1 = x2 - 1  # int32
                y2 = x1 * x3 + x2.astype(x1.dtype)
                y3 = x1 + x3
                self.retain_inputs((0, 2))
                # TODO(niboshi): Include 0 in retain_outputs to test integer
                # array retention
                self.retain_outputs((1,))
                return y1, y2, y3

            def backward(self, inputs, grad_outputs):
                x1, x2, x3 = inputs
                assert x2 is None  # not retained
                gy1, gy2, gy3 = grad_outputs
                assert gy1 is None  # y1 is int32

                assert len(self.output_data) == 3
                assert self.output_data[0] is None
                assert self.output_data[2] is None
                _, y2, _ = self.output_data
                assert isinstance(y2, backward_xp.ndarray)

                # y3 is disconnected
                # TODO(niboshi): Expression after "or" is workaround for
                # chainerx. ChainerX backward should return None for
                # disconnected output and this workaround should be removed.
                assert (gy3 is None
                        or (float(gy3.max()) == 0
                            and float((-gy3).max()) == 0))

                assert isinstance(x1, backward_xp.ndarray)
                assert isinstance(x3, backward_xp.ndarray)
                assert isinstance(gy2, backward_xp.ndarray)

                gx1 = x3 * gy2  # + gy3
                gx2 = None
                gx3 = x1 * gy2  # + gy3
                return gx1, gx2, gx3

        return Func()(x1, x2, x3)

    def call_func_function_node(self, backend_config, x1, x2, x3):
        xp = backend_config.xp

        if xp is chainerx:
            backend_name, device_index = (
                backend_config.chainerx_device.split(':'))
            if backend_name == 'native':
                forward_xp = numpy
            elif backend_name == 'cuda':
                forward_xp = cuda.cupy
            else:
                assert False
            backward_xp = chainerx
        else:
            forward_xp = xp
            backward_xp = xp

        class Func(chainer.FunctionNode):
            def forward(self, inputs):
                # x1, x3: float32
                # x2: int32
                x1, x2, x3 = inputs

                assert isinstance(x1, forward_xp.ndarray)
                assert isinstance(x2, forward_xp.ndarray)
                assert isinstance(x3, forward_xp.ndarray)

                y1 = x2 - 1  # int32
                y2 = x1 * x3 + x2.astype(x1.dtype)
                y3 = x1 + x3
                self.retain_inputs((0, 2))
                # TODO(niboshi): Include 0 in retain_outputs to test integer
                # array retention
                self.retain_outputs((1,))
                return y1, y2, y3

            def backward(self, input_indexes, grad_outputs):
                assert input_indexes == (0, 2)
                x1, x3 = self.get_retained_inputs()
                gy1, gy2, gy3 = grad_outputs
                assert gy1 is None  # y1 is int32

                y2, = self.get_retained_outputs()
                assert isinstance(y2.array, backward_xp.ndarray)

                # y3 is disconnected
                # TODO(niboshi): Expression after "or" is workaround for
                # chainerx. ChainerX backward should return None for
                # disconnected output and this workaround should be removed.
                assert (gy3 is None
                        or (float(gy3.array.max()) == 0
                            and float((-gy3.array).max()) == 0))

                assert isinstance(x1.array, backward_xp.ndarray)
                assert isinstance(x3.array, backward_xp.ndarray)
                assert isinstance(gy2.array, backward_xp.ndarray)

                gx1 = x3 * gy2  # + gy3
                gx2 = None
                gx3 = x1 * gy2  # + gy3
                return gx1, gx2, gx3

        return Func().apply((x1, x2, x3))

    def call_func(self, backend_config, x1, x2, x3):
        if self.function_node:
            return self.call_func_function_node(backend_config, x1, x2, x3)
        else:
            return self.call_func_function(backend_config, x1, x2, x3)

    def test_backprop(self, backend_config):

        x1_arr = numpy.array([2, 3], numpy.float32)
        x2_arr = numpy.array([3, 1], numpy.int32)
        x3_arr = numpy.array([5, 2], numpy.float32)
        gy2_arr = numpy.array([2, 4], numpy.float32)
        x1_arr, x2_arr, x3_arr, gy2_arr = backend_config.get_array(
            (x1_arr, x2_arr, x3_arr, gy2_arr))

        x1 = chainer.Variable(x1_arr)
        x2 = chainer.Variable(x2_arr, requires_grad=False)
        x3 = chainer.Variable(x3_arr)

        # Forward
        y1, y2, y3 = self.call_func(backend_config, x1, x2, x3)

        assert isinstance(y1.array, backend_config.xp.ndarray)
        assert isinstance(y2.array, backend_config.xp.ndarray)
        assert isinstance(y3.array, backend_config.xp.ndarray)

        # Backward
        y2.grad = gy2_arr
        y2.backward()

        assert isinstance(x1.grad, backend_config.xp.ndarray)
        assert x2.grad is None
        assert isinstance(x3.grad, backend_config.xp.ndarray)


testing.run_module(__name__, __file__)
