import unittest

import numpy
import pytest

import chainer
from chainer import initializers
from chainer import testing


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
        testing.test._check_contiguousness(arr, self.contiguous)


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

    param_names = ['p']

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

    def get_initializers(self):
        return [
            initializers.Constant(0), 2,
            testing.link.InitializerPair(None, initializers.Constant(1))],

    def create_link(self, initializers):
        initial_p, = initializers
        in_size = self.in_size
        out_size = self.out_size
        return DotLink(in_size, out_size, initial_p)

    def generate_inputs(self):
        return numpy.random.rand(self.n, self.in_size).astype(self.dtype),

    def forward_expected(self, inputs, params):
        x, = inputs
        p, = params
        return numpy.dot(x, p),


@_inject_backend_tests
class TestLinkCorrect(DotLinkTestBase, testing.LinkTestCase):
    pass


@_inject_backend_tests
@pytest.mark.xfail(strict=True, raises=TypeError)
class TestLinkInvalidDefaultInitializer(DotLinkTestBase, testing.LinkTestCase):

    skip_forward_test = True
    skip_backward_test = True

    default_initializer = chainer


@_inject_backend_tests
@pytest.mark.xfail(strict=True, raises=testing.LinkTestError)
class TestLinkIncorrectForward(DotLinkTestBase, testing.LinkTestCase):

    skip_backward_test = True
    skip_initializers_test = True

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
    skip_initializers_test = True

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
    skip_initializers_test = True

    def create_link(self, initializers):
        initial_p, = initializers
        in_size = self.in_size
        out_size = self.out_size
        link = DotLinkIncorrectBackward(
            False, True, in_size, out_size, initial_p)
        return link


@_inject_backend_tests
@pytest.mark.xfail(strict=True, raises=testing.LinkTestError)
class TestLinkIncorrectBackwardInitializers(
        DotLinkTestBase, testing.LinkTestCase):

    skip_forward_test = True
    skip_backward_test = True

    def create_link(self, initializers):
        initial_p, = initializers
        in_size = self.in_size
        out_size = self.out_size
        link = DotLinkIncorrectInitialization(in_size, out_size, initial_p)
        return link


@_inject_backend_tests
@pytest.mark.xfail(strict=True, raises=TypeError)
class TestLinkIncorrectCreateLink(DotLinkTestBase, testing.LinkTestCase):

    def create_link(self, initializers):
        # Invalid return type (that is not an instance of chainer.Link).
        return numpy.array([1])


@testing.parameterize(*testing.product({
    'invalid_initializer': [
        chainer.Variable(numpy.array([1])),
        chainer.Parameter(numpy.array([1])),
        testing.link.InitializerPair(None, None),
    ]}))
@_inject_backend_tests
@pytest.mark.xfail(strict=True, raises=TypeError)
class TestLinkIncorrectInitializers(DotLinkTestBase, testing.LinkTestCase):

    skip_forward_test = True
    skip_backward_test = True

    def get_initializers(self):
        return [self.invalid_initializer],


@testing.parameterize(*testing.product({
    'invalid_forward_backward_initializer': [
        chainer.Variable(numpy.array([1])),
        chainer.Parameter(numpy.array([1])),
        testing.link.InitializerPair(None, None),
    ]}))
@_inject_backend_tests
@pytest.mark.xfail(strict=True, raises=TypeError)
class TestLinkIncorrectForwardBackwardInitializers(
        DotLinkTestBase, testing.LinkTestCase):

    skip_initializers_test = True

    def generate_params(self):
        return self.invalid_forward_backward_initializer,


@_inject_backend_tests
class TestLinkOnlyInitializers(testing.LinkTestCase):

    """`generate_params` is not required if forward and
    backward tests are skipped.
    """

    skip_forward_test = True
    skip_backward_test = True

    param_names = ['p']

    def get_initializers(self):
        return [
            initializers.Constant(0), 2,
            testing.link.InitializerPair(None, initializers.Constant(1))],

    def create_link(self, initializers):
        initial_p, = initializers
        return DotLink(2, 3, initial_p)

    def generate_inputs(self):
        return numpy.random.rand(1, 2).astype(numpy.float32),


@_inject_backend_tests
class TestLinkOnlyForwardBackward(testing.LinkTestCase):

    """`get_initializers` is not required if initializers test is skipped."""

    skip_initializers_test = True

    param_names = ['p']

    def generate_params(self):
        return numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32),

    def create_link(self, initializers):
        initial_p, = initializers
        return DotLink(2, 3, initial_p)

    def generate_inputs(self):
        return numpy.random.rand(1, 2).astype(numpy.float32),

    def forward_expected(self, inputs, params):
        x, = inputs
        p, = params
        return numpy.dot(x, p),


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
@pytest.mark.xfail(strict=True, raises=testing.test._ContiguousnessMatched)
class TestLinkContiguousness(DotLinkTestBase, testing.LinkTestCase):

    skip_initializers_test = True

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
