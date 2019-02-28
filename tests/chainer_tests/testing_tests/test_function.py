import unittest

import numpy
import pytest

import chainer
from chainer import testing
from chainer import utils
import chainerx


_inject_backend_tests = testing.inject_backend_tests(
    None,
    [
        # CPU tests
        {},
        {'use_ideep': 'always'},
        # GPU tests
        {'use_cuda': True},
        {'use_cuda': True, 'cuda_device': 1},
        # ChainerX tests
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
    ])


def _forward_correct(x1, x2):
    dt = x1.dtype.type
    y1 = (x1 + x2) ** dt(2)
    y2 = (x1 ** dt(2)) * (x2 ** dt(2))
    return utils.force_array(y1), utils.force_array(y2)


def _backward_correct(x1, x2, gy1, gy2):
    dt = x1.dtype.type
    ggx1 = (
        + gy1 * dt(2) * (x1 + x2)
        + gy2 * dt(2) * x1 * x2 ** dt(2))
    ggx2 = (
        + gy1 * dt(2) * (x1 + x2)
        + gy2 * dt(2) * x1 ** dt(2) * x2)
    return ggx1, ggx2


def _double_backward_correct(x1, x2, gy1, gy2, ggx1, ggx2):
    dt = x1.dtype.type
    ggy1 = (ggx1 + ggx2) * dt(2) * (x1 + x2)
    ggy2 = (ggx1 * x2 + ggx2 * x1) * dt(2) * x1 * x2
    gx1 = (
        + ggx1 * (dt(2) * gy1 + dt(2) * x2 ** dt(2) * gy2)
        + ggx2 * (dt(2) * gy1 + dt(4) * x1 * x2 * gy2))
    gx2 = (
        + ggx1 * (dt(2) * gy1 + dt(4) * x1 * x2 * gy2)
        + ggx2 * (dt(2) * gy1 + dt(2) * x1 ** dt(2) * gy2))
    return gx1, gx2, ggy1, ggy2


# TestFunctionTestSuccessful
#
# This test checks for successfull case.
# Incoming array types are also checked.

class FuncCorrectlyImplemented(chainer.FunctionNode):
    def __init__(self, device):
        self.device = device

    def forward(self, inputs):
        device = self.device
        x1, x2 = inputs
        if device.xp is chainerx:
            fallback_device = device.fallback_device
            assert isinstance(x1, fallback_device.supported_array_types)
            assert isinstance(x2, fallback_device.supported_array_types)

        self.retain_inputs((0, 1))
        y1, y2 = _forward_correct(x1, x2)
        return utils.force_array(y1), utils.force_array(y2)

    def backward(self, indexes, grad_outputs):
        device = self.device
        x1, x2 = self.get_retained_inputs()
        gy1, gy2 = grad_outputs
        assert isinstance(x1.array, device.supported_array_types)
        assert isinstance(x2.array, device.supported_array_types)
        assert isinstance(gy1.array, device.supported_array_types)
        assert isinstance(gy2.array, device.supported_array_types)

        grad_func = FuncGradCorrectlyImplemented(device)
        return grad_func.apply((x1, x2, gy1, gy2))


class FuncGradCorrectlyImplemented(chainer.FunctionNode):
    def __init__(self, device):
        self.device = device

    def forward(self, inputs_and_grad_outputs):
        device = self.device
        x1, x2, gy1, gy2 = inputs_and_grad_outputs
        if device.xp is chainerx:
            fallback_device = device.fallback_device
            assert isinstance(gy1, fallback_device.supported_array_types)
            assert isinstance(gy2, fallback_device.supported_array_types)

        self.retain_inputs((0, 1, 2, 3))

        ggx1, ggx2 = _backward_correct(x1, x2, gy1, gy2)
        return utils.force_array(ggx1), utils.force_array(ggx2)

    def backward(self, indexes, grad_grad_inputs):
        device = self.device
        ggx1, ggx2 = grad_grad_inputs
        assert isinstance(ggx1, chainer.Variable)
        assert isinstance(ggx2, chainer.Variable)
        assert isinstance(ggx1.array, device.supported_array_types)
        assert isinstance(ggx2.array, device.supported_array_types)
        x1, x2, gy1, gy2 = self.get_retained_inputs()
        assert isinstance(x1, chainer.Variable)
        assert isinstance(x2, chainer.Variable)
        assert isinstance(gy1, chainer.Variable)
        assert isinstance(gy2, chainer.Variable)
        assert isinstance(x1.array, device.supported_array_types)
        assert isinstance(x2.array, device.supported_array_types)
        assert isinstance(gy1.array, device.supported_array_types)
        assert isinstance(gy2.array, device.supported_array_types)

        gx1, gx2, ggy1, ggy2 = _double_backward_correct(
            x1, x2, gy1, gy2, ggx1, ggx2)
        return gx1, gx2, ggy1, ggy2


@testing.parameterize(*testing.product({
    'shape': [(3, 2), (2,), (1,), (), (2, 0, 3)],
}))
@_inject_backend_tests
class TestFunctionTestSuccessful(testing.FunctionTestCase):

    def generate_inputs(self):
        x1 = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        x2 = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        return x1, x2

    def forward(self, inputs, device):
        func = FuncCorrectlyImplemented(device)
        return func.apply(inputs)

    def forward_expected(self, inputs):
        return _forward_correct(*inputs)


# TestFunctionTestIncorrectForward
#
# This test checks if it can detect incorrect forward implementation.

class FuncWithIncorrectForward(chainer.FunctionNode):
    def forward(self, inputs):
        x1, x2 = inputs
        y1, y2 = _forward_correct(x1, x2)
        y1, y2 = utils.force_array(y1), utils.force_array(y2)
        y2[...] += 1  # ! make incorrect
        return y1, y2

    def backward(self, *args, **kwargs):
        assert False  # should never be called


@testing.parameterize(*testing.product({
    'shape': [(3, 2), (2,), (1,), ()],
}))
@_inject_backend_tests
@pytest.mark.xfail(strict=True, raises=testing.FunctionTestError)
class TestFunctionTestIncorrectForward(testing.FunctionTestCase):
    skip_backward_test = True
    skip_double_backward_test = True

    def generate_inputs(self):
        x1 = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        x2 = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        return x1, x2

    def forward(self, inputs, device):
        func = FuncWithIncorrectForward()
        return func.apply(inputs)

    def forward_expected(self, inputs):
        return _forward_correct(*inputs)


# TestFunctionTestIncorrectBackward
#
# This test checks if it can detect incorrect backward implementation.

class FuncWithIncorrectBackward(chainer.FunctionNode):
    def forward(self, inputs):
        x1, x2 = inputs
        y1, y2 = _forward_correct(x1, x2)
        self.retain_inputs((0, 1))
        return utils.force_array(y1), utils.force_array(y2)

    def backward(self, indexes, grad_outputs):
        gy1, gy2 = grad_outputs
        x1, x2 = self.get_retained_inputs()
        ggx1, ggx2 = _backward_correct(x1, x2, gy1, gy2)
        ggx2 = ggx2 + 10000  # ! make incorrect
        return utils.force_array(ggx1), utils.force_array(ggx2)


@testing.parameterize(*testing.product({
    'shape': [(3, 2), (2,), (1,), ()],
}))
@_inject_backend_tests
@pytest.mark.xfail(strict=True, raises=testing.FunctionTestError)
class TestFunctionTestIncorrectBackward(testing.FunctionTestCase):
    skip_forward_test = True
    skip_double_backward_test = True

    def generate_inputs(self):
        x1 = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        x2 = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        return x1, x2

    def forward(self, inputs, device):
        func = FuncWithIncorrectBackward()
        return func.apply(inputs)

    def forward_expected(self, inputs):
        return _forward_correct(*inputs)


# TestFunctionTestIncorrectDoubleBackward
#
# This test checks if it can detect incorrect double backward implementation.

class FuncWithIncorrectDoubleBackward(chainer.FunctionNode):
    def forward(self, inputs):
        x1, x2 = inputs
        y1, y2 = _forward_correct(x1, x2)
        self.retain_inputs((0, 1))
        return utils.force_array(y1), utils.force_array(y2)

    def backward(self, indexes, grad_outputs):
        x1, x2 = self.get_retained_inputs()
        gy1, gy2 = grad_outputs
        grad_func = FuncGradWithIncorrectDoubleBackward()
        return grad_func.apply((x1, x2, gy1, gy2))


class FuncGradWithIncorrectDoubleBackward(chainer.FunctionNode):
    def forward(self, inputs_and_grad_outputs):
        x1, x2, gy1, gy2 = inputs_and_grad_outputs
        self.retain_inputs((0, 1, 2, 3))

        ggx1, ggx2 = _backward_correct(x1, x2, gy1, gy2)
        return utils.force_array(ggx1), utils.force_array(ggx2)

    def backward(self, indexes, grad_grad_inputs):
        ggx1, ggx2 = grad_grad_inputs
        x1, x2, gy1, gy2 = self.get_retained_inputs()
        gx1, gx2, ggy1, ggy2 = _double_backward_correct(
            x1, x2, gy1, gy2, ggx1, ggx2)
        ggy2 = ggy2 + 10000  # ! make incorrect
        return gx1, gx2, ggy1, ggy2


@testing.parameterize(*testing.product({
    'shape': [(3, 2), (2,), (1,), ()],
}))
@_inject_backend_tests
@pytest.mark.xfail(strict=True, raises=testing.FunctionTestError)
class TestFunctionTestIncorrectDoubleBackward(testing.FunctionTestCase):
    skip_forward_test = True
    skip_backward_test = True

    def generate_inputs(self):
        x1 = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        x2 = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        return x1, x2

    def forward(self, inputs, device):
        func = FuncWithIncorrectDoubleBackward()
        return func.apply(inputs)

    def forward_expected(self, inputs):
        return _forward_correct(*inputs)


# FunctionTestCaseArrayContiguousnessTest


class FuncWithContiguousnessCheck(chainer.FunctionNode):
    def __init__(self, contiguous, check_on):
        self.contiguous = contiguous
        self.check_on = check_on

    def _check_contiguousness(self, arr):
        assert isinstance(arr, chainer.get_array_types())
        testing.test._check_contiguousness(arr, self.contiguous)

    def forward(self, inputs):
        x1, x2 = inputs
        if self.check_on == 'forward_input':
            self._check_contiguousness(x1)
            self._check_contiguousness(x2)

        self.retain_inputs((0, 1))
        y1, y2 = _forward_correct(x1, x2)
        return utils.force_array(y1), utils.force_array(y2)

    def backward(self, indexes, grad_outputs):
        x1, x2 = self.get_retained_inputs()
        gy1, gy2 = grad_outputs
        if self.check_on == 'backward_retained_input':
            self._check_contiguousness(x1.array)
            self._check_contiguousness(x2.array)
        elif self.check_on == 'backward_grad_output':
            self._check_contiguousness(gy1.array)
            self._check_contiguousness(gy2.array)

        grad_func = FuncGradWithContiguousnessCheck(
            self.contiguous, self.check_on)
        return grad_func.apply((x1, x2, gy1, gy2))


class FuncGradWithContiguousnessCheck(chainer.FunctionNode):
    def __init__(self, contiguous, check_on):
        self.contiguous = contiguous
        self.check_on = check_on

    def _check_contiguousness(self, arr):
        testing.test._check_contiguousness(arr, self.contiguous)

    def forward(self, inputs_and_grad_outputs):
        x1, x2, gy1, gy2 = inputs_and_grad_outputs
        self.retain_inputs((0, 1, 2, 3))

        ggx1, ggx2 = _backward_correct(x1, x2, gy1, gy2)
        return utils.force_array(ggx1), utils.force_array(ggx2)

    def backward(self, indexes, grad_grad_inputs):
        ggx1, ggx2 = grad_grad_inputs
        if self.check_on == 'double_backward_grad_grad_input':
            self._check_contiguousness(ggx1)
            self._check_contiguousness(ggx2)
        x1, x2, gy1, gy2 = self.get_retained_inputs()

        gx1, gx2, ggy1, ggy2 = _double_backward_correct(
            x1, x2, gy1, gy2, ggx1, ggx2)
        return gx1, gx2, ggy1, ggy2


@testing.parameterize(*testing.product({
    'shape': [(3, 2), (2,), (1, 2)],
    'contiguous': [None, 'C'],
    'check_on': [  # Check points in which cotiguousness is probed.
        'forward_input',
        # TODO(niboshi): As gradient_check.check_backward currently copies the
        # grads without preserving strides, they cannot be non-contiguous.
        # Enable this check after check_backward will be fixed.
        # 'backward_grad_output',
        'backward_retained_input',
        # TODO(niboshi): Enable this check after check_backward will be fixed.
        # 'double_backward_grad_grad_input',
    ]}))
@_inject_backend_tests
@pytest.mark.xfail(strict=True, raises=testing.test._ContiguousnessMatched)
class FunctionTestCaseArrayContiguousnessTest(testing.FunctionTestCase):
    def generate_inputs(self):
        x1 = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        x2 = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        return x1, x2

    def forward(self, inputs, device):
        func = FuncWithContiguousnessCheck(self.contiguous, self.check_on)
        return func.apply(inputs)

    def forward_expected(self, inputs):
        return _forward_correct(*inputs)

    def before_test(self, test_name):
        # Some combinations of test methods and check points are irrelevant.
        # Skip such combinations.
        # For example, `test_forward` method does not generate grad_outputs.
        if test_name == 'test_forward':
            if self.check_on != 'forward_input':
                raise unittest.SkipTest()
        if test_name == 'test_backward':
            if self.check_on == 'double_backward_grad_grad_input':
                raise unittest.SkipTest()


testing.run_module(__name__, __file__)
