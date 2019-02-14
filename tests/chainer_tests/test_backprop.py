import unittest

import numpy

import chainer
from chainer import testing
import chainer.testing.backend
import chainerx


def _get_expected_xp(backend_config, is_function):
    # Returns a pair of xp's expected in forward() and backward() respectively.
    xp = backend_config.xp

    if xp is chainerx:
        forward_xp = backend_config.device.fallback_device.xp
    else:
        forward_xp = xp

    if is_function:
        # chainer.Function
        backward_xp = forward_xp
    else:
        # chainer.FunctionNode
        backward_xp = xp

    return forward_xp, backward_xp


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
        forward_xp, backward_xp = _get_expected_xp(backend_config, True)

        class Func(chainer.Function):
            def __init__(self):
                self.array_init = backend_config.device.send(
                    numpy.array([3], numpy.float32))

            def forward(self, inputs):
                # Inputs
                assert isinstance(inputs, tuple)
                # x1, x3: float32
                # x2: int32
                x1, x2, x3 = inputs
                assert isinstance(x1, forward_xp.ndarray)
                assert isinstance(x2, forward_xp.ndarray)
                assert isinstance(x3, forward_xp.ndarray)

                # attribute fallback
                assert isinstance(self.array_init, forward_xp.ndarray)
                self.array_forward = forward_xp.array([2], numpy.float32)
                assert isinstance(self.array_forward, forward_xp.ndarray)

                y1 = x2 - 1  # int32
                y2 = x1 * x3 + x2.astype(x1.dtype)
                y3 = x1 + x3
                self.retain_inputs((0, 2))
                self.retain_outputs((0, 1,))
                return y1, y2, y3

            def backward(self, inputs, grad_outputs):

                # Retained inputs
                assert isinstance(inputs, tuple)
                x1, x2, x3 = inputs
                assert isinstance(x1, backward_xp.ndarray)
                assert x2 is None  # not retained
                assert isinstance(x3, backward_xp.ndarray)

                # Output gradients
                assert isinstance(grad_outputs, tuple)
                gy1, gy2, gy3 = grad_outputs
                assert gy1 is None  # y1 is int32
                # y3 is disconnected
                # TODO(niboshi): Expression after "or" is workaround for
                # chainerx. ChainerX backward should return None for
                # disconnected output and this workaround should be removed.
                assert (gy3 is None
                        or (float(gy3.max()) == 0
                            and float((-gy3).max()) == 0))

                # Retained outputs
                output_data = self.output_data
                assert isinstance(output_data, tuple)
                y1, y2, y3 = output_data
                assert isinstance(y1, backward_xp.ndarray)
                assert isinstance(y2, backward_xp.ndarray)
                assert y3 is None

                # attribute fallback
                assert isinstance(self.array_init, backward_xp.ndarray)
                assert isinstance(self.array_forward, backward_xp.ndarray)
                self.array_backward = backward_xp.array([4], numpy.float32)
                assert isinstance(self.array_backward, backward_xp.ndarray)

                gx1 = x3 * gy2  # + gy3
                gx2 = None
                gx3 = x1 * gy2  # + gy3
                return gx1, gx2, gx3

        return Func()(x1, x2, x3)

    def call_func_function_node(self, backend_config, x1, x2, x3):
        forward_xp, backward_xp = _get_expected_xp(backend_config, False)

        class Func(chainer.FunctionNode):
            def __init__(self):
                self.array_init = backend_config.device.send(
                    numpy.array([3], numpy.float32))

            def forward(self, inputs):

                # Inputs
                # x1, x3: float32
                # x2: int32
                x1, x2, x3 = inputs
                assert isinstance(x1, forward_xp.ndarray)
                assert isinstance(x2, forward_xp.ndarray)
                assert isinstance(x3, forward_xp.ndarray)

                # attribute fallback
                assert isinstance(self.array_init, forward_xp.ndarray)
                self.array_forward = forward_xp.array([2], numpy.float32)
                assert isinstance(self.array_forward, forward_xp.ndarray)

                y1 = x2 - 1  # int32
                y2 = x1 * x3 + x2.astype(x1.dtype)
                y3 = x1 + x3
                self.retain_inputs((0, 2))
                self.retain_outputs((0, 1,))
                return y1, y2, y3

            def backward(self, input_indexes, grad_outputs):

                # Input indexes
                assert isinstance(input_indexes, tuple)
                assert input_indexes == (0, 2)

                # Retained inputs
                retained_inputs = self.get_retained_inputs()
                assert isinstance(retained_inputs, tuple)
                x1, x3 = retained_inputs
                assert isinstance(x1.array, backward_xp.ndarray)
                assert isinstance(x3.array, backward_xp.ndarray)

                # Output gradients
                assert isinstance(grad_outputs, tuple)
                gy1, gy2, gy3 = grad_outputs
                assert gy1 is None  # y1 is int32
                assert isinstance(gy2.array, backward_xp.ndarray)
                # y3 is disconnected
                # TODO(niboshi): Expression after "or" is workaround for
                # chainerx. ChainerX backward should return None for
                # disconnected output and this workaround should be removed.
                assert (gy3 is None
                        or (float(gy3.array.max()) == 0
                            and float((-gy3.array).max()) == 0))

                # Retained outputs
                retained_outputs = self.get_retained_outputs()
                assert isinstance(retained_outputs, tuple)
                y1, y2, = retained_outputs
                assert isinstance(y1.array, backward_xp.ndarray)
                assert isinstance(y2.array, backward_xp.ndarray)

                # attribute fallback
                assert isinstance(self.array_init, backward_xp.ndarray)
                assert isinstance(self.array_forward, backward_xp.ndarray)
                self.array_backward = backward_xp.array([4], numpy.float32)
                assert isinstance(self.array_backward, backward_xp.ndarray)

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
class TestFunctionInputNone(unittest.TestCase):

    def call_func_function(self, backend_config, x2):
        forward_xp, backward_xp = _get_expected_xp(backend_config, True)

        class Func(chainer.Function):
            def forward(self, inputs):

                # Inputs
                assert isinstance(inputs, tuple)
                x1, x2, x3 = inputs
                assert x1 is None
                assert isinstance(x2, forward_xp.ndarray)
                assert x3 is None

                y1 = x2 * 3
                self.retain_inputs((1, 2))
                self.retain_outputs(())
                return y1,

            def backward(self, inputs, grad_outputs):

                # Retained inputs
                assert isinstance(inputs, tuple)
                x1, x2, x3 = inputs
                assert x1 is None
                assert isinstance(x2, backward_xp.ndarray)
                assert x3 is None

                # Output gradients
                assert isinstance(grad_outputs, tuple)
                gy1, = grad_outputs
                assert isinstance(gy1, backward_xp.ndarray)

                # Retained outputs
                output_data = self.output_data
                assert isinstance(output_data, tuple)
                y1, = output_data
                assert y1 is None

                gx2 = 3 * gy1
                return None, gx2, None

        return Func()(None, x2, None),

    def call_func_function_node(self, backend_config, x2):
        forward_xp, backward_xp = _get_expected_xp(backend_config, False)

        class Func(chainer.FunctionNode):
            def forward(self, inputs):

                # Inputs
                x1, x2, x3 = inputs
                assert x1 is None
                assert isinstance(x2, forward_xp.ndarray)
                assert x3 is None

                y1 = x2 * 3
                self.retain_inputs((1, 2))
                self.retain_outputs(())
                return y1,

            def backward(self, input_indexes, grad_outputs):

                # Input indexes
                assert isinstance(input_indexes, tuple)
                assert input_indexes == (1,)

                # Retained inputs
                retained_inputs = self.get_retained_inputs()
                assert isinstance(retained_inputs, tuple)
                x2, x3 = retained_inputs
                assert isinstance(x2.array, backward_xp.ndarray)
                assert x3 is None

                # Output grads
                assert isinstance(grad_outputs, tuple)
                gy1, = grad_outputs
                assert isinstance(gy1.array, backward_xp.ndarray)

                # Retained outputs
                retained_outputs = self.get_retained_outputs()
                assert retained_outputs is ()

                gx2 = 3 * gy1
                return None, gx2, None

        return Func().apply((None, x2, None))

    def call_func(self, backend_config, x1):
        if self.function_node:
            return self.call_func_function_node(backend_config, x1)
        else:
            return self.call_func_function(backend_config, x1)

    def test_backprop(self, backend_config):

        x2_arr = numpy.array([2, 3], numpy.float32)
        gy1_arr = numpy.array([2, 4], numpy.float32)
        x2_arr, gy1_arr = backend_config.get_array((x2_arr, gy1_arr))

        x2 = chainer.Variable(x2_arr, requires_grad=True)

        # Forward
        y1, = self.call_func(backend_config, x2)

        assert isinstance(y1.array, backend_config.xp.ndarray)

        # Backward
        y1.grad = gy1_arr
        y1.backward()

        assert isinstance(x2.grad, backend_config.xp.ndarray)


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
class TestFunctionOutputNone(unittest.TestCase):

    def call_func_function(self, backend_config, x1):
        forward_xp, backward_xp = _get_expected_xp(backend_config, True)

        class Func(chainer.Function):
            def forward(self, inputs):

                # Inputs
                assert isinstance(inputs, tuple)
                x1, = inputs
                assert isinstance(x1, forward_xp.ndarray)

                y2 = x1 * 3 + 2
                self.retain_inputs(())
                self.retain_outputs((1, 2,))
                return None, y2, None

            def backward(self, inputs, grad_outputs):

                # Retained inputs
                assert isinstance(inputs, tuple)
                x1, = inputs
                assert x1 is None

                # Output gradients
                assert isinstance(grad_outputs, tuple)
                gy1, gy2, gy3 = grad_outputs
                assert gy1 is None
                assert isinstance(gy2, backward_xp.ndarray)
                assert gy3 is None

                # Retained outputs
                output_data = self.output_data
                assert isinstance(output_data, tuple)
                assert len(output_data) == 3
                y1, y2, y3 = output_data
                assert y1 is None
                assert isinstance(y2, backward_xp.ndarray)
                assert y3 is None

                gx1 = 3 * gy2
                return gx1,

        return Func()(x1)

    def call_func_function_node(self, backend_config, x1):
        forward_xp, backward_xp = _get_expected_xp(backend_config, False)

        class Func(chainer.FunctionNode):
            def forward(self, inputs):

                # Inputs
                x1, = inputs
                assert isinstance(x1, forward_xp.ndarray)

                y2 = x1 * 3 + 2
                self.retain_outputs((1, 2))
                return None, y2, None

            def backward(self, input_indexes, grad_outputs):

                # Input indexes
                assert isinstance(input_indexes, tuple)
                assert input_indexes == (0,)

                # Retained inputs
                retained_inputs = self.get_retained_inputs()
                assert isinstance(retained_inputs, tuple)
                assert retained_inputs == ()

                # Output grads
                assert isinstance(grad_outputs, tuple)
                gy1, gy2, gy3 = grad_outputs
                assert gy1 is None
                assert isinstance(gy2.array, backward_xp.ndarray)
                assert gy3 is None

                # Retained outputs
                retained_outputs = self.get_retained_outputs()
                assert isinstance(retained_outputs, tuple)
                y2, y3 = retained_outputs
                assert y3 is None
                assert isinstance(y2.array, backward_xp.ndarray)

                gx1 = 3 * gy2
                return gx1,

        return Func().apply((x1,))

    def call_func(self, backend_config, x1):
        if self.function_node:
            return self.call_func_function_node(backend_config, x1)
        else:
            return self.call_func_function(backend_config, x1)

    def test_backprop(self, backend_config):

        x1_arr = numpy.array([2, 3], numpy.float32)
        gy2_arr = numpy.array([2, 4], numpy.float32)
        x1_arr, gy2_arr = backend_config.get_array((x1_arr, gy2_arr))

        x1 = chainer.Variable(x1_arr, requires_grad=True)

        # Forward
        y1, y2, y3 = self.call_func(backend_config, x1)

        assert y1.array is None
        assert isinstance(y2.array, backend_config.xp.ndarray)
        assert y3.array is None

        # Backward
        y2.grad = gy2_arr
        y2.backward()

        assert isinstance(x1.grad, backend_config.xp.ndarray)


testing.run_module(__name__, __file__)
