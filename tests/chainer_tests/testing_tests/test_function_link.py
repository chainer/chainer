import unittest

import numpy
import pytest

import chainer
from chainer import initializers
from chainer import testing
from chainer import utils
import chainerx


# Utilities for contiguousness tests.
#
# These tests checks incoming array contiguousness.
# As it's not possible to assume contiguousness of incoming arrays consistently
# (because gradient_check passes contiguous arrays in numerical_grad),
# we instead simulate the test failure. The function implementation raises an
# error if an incoming array matches the expected contiguousness and we expect
# the failure.

class _ContiguousnessMatched(Exception):
    pass


def _is_f_contiguous(shape, strides, itemsize):
    if numpy.prod(shape) <= 1:
        return True
    for sh, st in zip(shape, reversed(strides)):
        if sh == 1:
            continue
        if st != itemsize:
            return False
        itemsize *= sh
    return True


def _get_contiguousness(arr):
    if isinstance(arr, chainerx.ndarray):
        c_contig = arr.is_contiguous
        f_contig = _is_f_contiguous(
            arr.shape, arr.strides, arr.itemsize)
        return (c_contig, f_contig)
    return (arr.flags.c_contiguous, arr.flags.f_contiguous)


def _check_contiguousness(arr, expected_contiguous):
    if isinstance(arr, chainer.Variable):
        _check_contiguousness(arr.array, expected_contiguous)
        return

    c_contig, f_contig = _get_contiguousness(arr)
    if numpy.prod(arr.shape) <= 1:
        return  # not applicable for this shape

    if expected_contiguous is None:
        # expected to be non-contiguous
        if not c_contig and not f_contig:
            raise _ContiguousnessMatched()
    elif expected_contiguous == 'C':
        # expected to be C-contiguous
        if c_contig:
            raise _ContiguousnessMatched()
    else:
        assert False


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
        _check_contiguousness(arr, self.contiguous)

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
        testing.function_link._check_contiguousness(arr, self.contiguous)

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
@pytest.mark.xfail(strict=True, raises=_ContiguousnessMatched)
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


class Dot(chainer.FunctionNode):

    def __init__(
            self, incorrect_forward=False, incorrect_backward_gx=False,
            incorrect_backward_gp=False, contiguous=None,
            check_on=None):
        self.incorrect_forward = incorrect_forward
        self.incorrect_backward_gx = incorrect_backward_gx
        self.incorrect_backward_gp = incorrect_backward_gp
        self.contiguous = contiguous
        self.check_on = check_on

    def forward(self, inputs):
        self.retain_inputs((0, 1))
        xp = chainer.backend.get_array_module(*inputs)
        x, p = inputs

        if self.check_on == 'forward_input':
            self._check_contiguousness(x)
            self._check_contiguousness(p)

        y = xp.dot(x, p)

        if self.incorrect_forward:
            y *= 9999

        return y,

    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs
        x, p = self.get_retained_inputs()

        if self.check_on == 'backward_retained_input':
            self._check_contiguousness(x.array)
            self._check_contiguousness(p.array)
        elif self.check_on == 'backward_grad_output':
            self._check_contiguousness(gy.array)

        gx = chainer.functions.matmul(gy, p.T)
        gp = chainer.functions.matmul(x.T, gy)

        if self.incorrect_backward_gx:
            gx /= 2
        if self.incorrect_backward_gp:
            gp += 1000

        return gx, gp

    def _check_contiguousness(self, arr):
        assert isinstance(arr, chainer.get_array_types())
        _check_contiguousness(arr, self.contiguous)


class DotLink(chainer.Link):

    """correctly implemented dot."""

    def __init__(
            self, in_size, out_size, initial_p=None, contiguous=None,
            check_on=None):
        super(DotLink, self).__init__()

        with self.init_scope():
            if initial_p is None:
                initial_p = initializers.Constant(1)
            self.p = chainer.Parameter(initial_p, shape=(in_size, out_size))

        self.contiguous = contiguous
        self.check_on = check_on

    def forward(self, inputs):
        x = inputs
        p = self.p
        contiguous = self.contiguous
        check_on = self.check_on
        y, = Dot(contiguous=contiguous, check_on=check_on).apply((x, p))
        return y


class DotLinkIncorrectForward(DotLink):

    """Incorrectly implemented dot (forward)."""

    def __init__(self, *args, **kwargs):
        super(DotLinkIncorrectForward, self).__init__(*args, **kwargs)

    def forward(self, inputs):
        x = inputs
        p = self.p
        y, = Dot(incorrect_forward=True).apply((x, p))
        return y


class DotLinkIncorrectBackward(DotLink):

    """Incorrect implementation of dot (backward)."""

    def __init__(self, incorrect_gx, incorrect_gp, *args, **kwargs):
        super(DotLinkIncorrectBackward, self).__init__(*args, **kwargs)
        self.incorrect_gx = incorrect_gx
        self.incorrect_gp = incorrect_gp

    def forward(self, inputs):
        x = inputs
        p = self.p
        y, = Dot(
            incorrect_backward_gx=self.incorrect_gx,
            incorrect_backward_gp=self.incorrect_gp).apply((x, p))
        return y


class DotLinkIncorrectInitialization(DotLink):

    """Incorrect implementation of dot (parameter initialization)."""

    def __init__(self, in_size, out_size, initial_p=None):
        # Ignores given initializer here.
        super(DotLinkIncorrectInitialization, self).__init__(
            in_size, out_size, initializers.Constant(0))


class DotLinkTestBase(object):

    param_names = ('p',)

    def setUp(self):
        self.n = 1
        self.in_size = 2
        self.out_size = 3
        self.dtype = numpy.float32

    def generate_params(self):
        in_size = self.in_size
        out_size = self.out_size
        return numpy.random.uniform(
            -1, 1, (in_size, out_size)).astype(self.dtype),

    def create_link(self, initializers):
        initial_p, = initializers
        in_size = self.in_size
        out_size = self.out_size
        return DotLink(in_size, out_size, initial_p)

    def generate_inputs(self):
        return numpy.random.rand(self.n, self.in_size).astype(self.dtype),

    # Required for forward backward tests.
    def forward_expected(self, link, inputs):
        p = link.p.array
        x, = inputs
        return numpy.dot(x, p),

    # Requires for initializers test.
    def get_initializers(self):
        return [
            initializers.Constant(0), 2,
            testing.InitializerArgument(None, initializers.Constant(1))],


@_inject_backend_tests
class TestLinkCorrect(DotLinkTestBase, testing.LinkTestCase):

    pass


@_inject_backend_tests
class TestLinkInitializersCorrect(
        DotLinkTestBase, testing.LinkInitializersTestCase):

    pass


@_inject_backend_tests
@pytest.mark.xfail(strict=True, raises=testing.LinkTestError)
class TestLinkIncorrectForward(DotLinkTestBase, testing.LinkTestCase):

    skip_backward_test = True

    def create_link(self, initializers):
        initial_p, = initializers
        in_size = self.in_size
        out_size = self.out_size
        link = DotLinkIncorrectForward(in_size, out_size, initial_p)
        return link


@_inject_backend_tests
@pytest.mark.xfail(strict=True, raises=testing.LinkTestError)
class TestLinkIncorrectBackwardInput(DotLinkTestBase, testing.LinkTestCase):

    skip_forward_test = True

    def create_link(self, initializers):
        initial_p, = initializers
        in_size = self.in_size
        out_size = self.out_size
        link = DotLinkIncorrectBackward(
            True, False, in_size, out_size, initial_p)
        return link


@_inject_backend_tests
@pytest.mark.xfail(strict=True, raises=testing.LinkTestError)
class TestLinkIncorrectBackwardParam(DotLinkTestBase, testing.LinkTestCase):

    skip_forward_test = True

    def create_link(self, initializers):
        initial_p, = initializers
        in_size = self.in_size
        out_size = self.out_size
        link = DotLinkIncorrectBackward(
            False, True, in_size, out_size, initial_p)
        return link


@_inject_backend_tests
@pytest.mark.xfail(strict=True, raises=TypeError)
class TestLinkIncorrectCreateLink(DotLinkTestBase, testing.LinkTestCase):

    def create_link(self, initializers):
        # Invalid return type (that is not an instance of chainer.Link).
        return numpy.array([1])


@testing.parameterize(*testing.product({
    'invalid_forward_backward_initializer': [
        chainer.Variable(numpy.array([1])),
        chainer.Parameter(numpy.array([1])),
    ]}))
@_inject_backend_tests
@pytest.mark.xfail(strict=True, raises=TypeError)
class TestLinkIncorrectForwardBackwardInitializers(
        DotLinkTestBase, testing.LinkTestCase):

    def generate_params(self):
        return self.invalid_forward_backward_initializer,


@_inject_backend_tests
@pytest.mark.xfail(strict=True, raises=testing.LinkTestError)
class TestLinkIncorrectBackwardInitializers(
        DotLinkTestBase, testing.LinkInitializersTestCase):

    def create_link(self, initializers):
        initial_p, = initializers
        in_size = self.in_size
        out_size = self.out_size
        link = DotLinkIncorrectInitialization(in_size, out_size, initial_p)
        return link


@testing.parameterize(*testing.product({
    'invalid_initializer': [
        chainer.Variable(numpy.array([1])),
        chainer.Parameter(numpy.array([1])),
    ]}))
@_inject_backend_tests
@pytest.mark.xfail(strict=True, raises=TypeError)
class TestLinkIncorrectInitializers(
        DotLinkTestBase, testing.LinkInitializersTestCase):

    def get_initializers(self):
        return [self.invalid_initializer],


@testing.parameterize(*testing.product({
    'contiguous': [None, 'C'],
    'check_on': [  # Check points in which cotiguousness is probed.
        'forward_input',
        # TODO(hvy): As gradient_check.check_backward currently copies the
        # grads without preserving strides, they cannot be non-contiguous.
        # Enable this check after check_backward will be fixed.
        # 'backward_grad_output',
        'backward_retained_input',
        # TODO(hvy): Enable this check after check_backward will be fixed.
        # 'double_backward_grad_grad_input',
    ]}))
@_inject_backend_tests
@pytest.mark.xfail(strict=True, raises=_ContiguousnessMatched)
class TestLinkContiguousness(DotLinkTestBase, testing.LinkTestCase):

    def before_test(self, test_name):
        # Some combinations of test methods and check points are irrelevant.
        # Skip such combinations.
        # For example, `test_forward` method does not generate grad_outputs.
        if test_name == 'test_forward':
            if self.check_on != 'forward_input':
                raise unittest.SkipTest()

    def create_link(self, initializers):
        initial_p, = initializers
        in_size = self.in_size
        out_size = self.out_size
        contiguous = self.contiguous
        check_on = self.check_on
        link = DotLink(
            in_size, out_size, initial_p, contiguous=contiguous,
            check_on=check_on)
        return link


testing.run_module(__name__, __file__)
