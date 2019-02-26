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


class DotLink(chainer.Link):

    # Correctly implemented dot.

    def __init__(self, in_size, out_size, initial_p=None):
        super(DotLink, self).__init__()

        with self.init_scope():
            if initial_p is None:
                initial_p = initializers.Constant(1)
            self.p = chainer.Parameter(initial_p, shape=(in_size, out_size))

    def forward(self, inputs):
        x = inputs
        p = self.p
        y = chainer.functions.matmul(x, p)
        return y


class IncorrectDot(chainer.FunctionNode):

    def __init__(
            self, incorrect_forward=False, incorrect_backward_gx=False,
            incorrect_backward_gp=False):
        self.incorrect_forward = incorrect_forward
        self.incorrect_backward_gx = incorrect_backward_gx
        self.incorrect_backward_gp = incorrect_backward_gp

    def forward(self, inputs):
        self.retain_inputs((0, 1))
        xp = chainer.backend.get_array_module(*inputs)
        x, p = inputs
        y = xp.dot(x, p)

        if self.incorrect_forward:
            y *= 9999

        return y,

    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs
        x, p = self.get_retained_inputs()
        gx = chainer.functions.matmul(gy, p.T)
        gp = chainer.functions.matmul(x.T, gy)

        if self.incorrect_backward_gx:
            gx /= 2
        if self.incorrect_backward_gp:
            gp += 1000

        return gx, gp


class DotLinkIncorrectForward(DotLink):

    # Incorrectly implemented dot (forward).

    def __init__(self, *args, **kwargs):
        super(DotLinkIncorrectForward, self).__init__(*args, **kwargs)

    def forward(self, inputs):
        x = inputs
        p = self.p
        y, = IncorrectDot(incorrect_forward=True).apply((x, p))
        return y


class DotLinkIncorrectBackward(DotLink):

    # Incorrect implementation of dot (backward).

    def __init__(self, incorrect_gx, incorrect_gp, *args, **kwargs):
        super(DotLinkIncorrectBackward, self).__init__(*args, **kwargs)
        self.incorrect_gx = incorrect_gx
        self.incorrect_gp = incorrect_gp

    def forward(self, inputs):
        x = inputs
        p = self.p
        y, = IncorrectDot(
            incorrect_backward_gx=self.incorrect_gx,
            incorrect_backward_gp=self.incorrect_gp).apply((x, p))
        return y


class DotLinkIncorrectInitialization(DotLink):

    # Incorrect implementation of dot (parameter initialization).

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

    def generate_forward_backward_initializers(self):
        in_size = self.in_size
        out_size = self.out_size
        return numpy.random.uniform(
            -1, 1, (in_size, out_size)).astype(self.dtype),

    def generate_initializers(self):
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

    def generate_initializers(self):
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

    def generate_forward_backward_initializers(self):
        return self.invalid_forward_backward_initializer,


@_inject_backend_tests
class TestLinkOnlyInitializers(testing.LinkTestCase):

    # `generate_forward_backward_initializers` is not required if forward and
    # backward tests are skipped.

    skip_forward_test = True
    skip_backward_test = True

    param_names = ['p']

    def generate_initializers(self):
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

    # `generate_initializers` is not required if initializers test is skipped.

    skip_initializers_test = True

    param_names = ['p']

    def generate_forward_backward_initializers(self):
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


testing.run_module(__name__, __file__)
