import numpy

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


class SimpleLinear(chainer.Link):

    def __init__(
            self, in_size, out_size=None, initial_W=None, initial_b=None):
        super(SimpleLinear, self).__init__()

        if out_size is None:
            in_size, out_size = None, in_size
        self.out_size = out_size

        with self.init_scope():
            if initial_W is None:
                initial_W = initializers.Constant(1)
            if initial_b is None:
                initial_b = initializers.Constant(0)
            self.W = chainer.Parameter(initial_W)
            self.b = chainer.Parameter(initial_b, shape=out_size)

        self.is_W_initialized = False
        if in_size is not None:
            self.initialize_W(in_size)

    def initialize_W(self, in_size):
        self.W.initialize((self.out_size, in_size))
        self.is_W_initialized = True

    def forward(self, inputs):
        if not self.is_W_initialized:
            self.initialize_W(inputs.shape[1])

        x = inputs
        W, b = self.W, self.b
        h = chainer.functions.matmul(x, W, transb=True)
        y = chainer.functions.add(h, b)
        return y


@_inject_backend_tests
@testing.parameterize(*(testing.product_dict(
    testing.product({
        'in_size': [None, 3],
        'out_size': [4],
    }),
)))
class TestLinkCorrectlyInitializedTest(testing.LinkTestCase):

    skip_forward_test = True
    skip_backward_test = True

    param_names = ['W', 'b']

    dtype = numpy.float32

    def generate_initializers(self):
        initial_W = [
            testing.link.InitializerPair(None, initializers.Constant(0))]
        initial_b = [
            initializers.Constant(2), 2,
            testing.link.InitializerPair(None, 0)]
        return initial_W, initial_b

    def create_link(self, initializers):
        initial_W, initial_b = initializers

        link = SimpleLinear(
            in_size=self.in_size,
            out_size=self.out_size,
            initial_W=initial_W,
            initial_b=initial_b)
        return link

    def generate_inputs(self):
        in_size = self.in_size
        if in_size is None:
            in_size = 1

        x = numpy.random.uniform(-1, 1, (3, in_size)).astype(self.dtype)
        return x,


testing.run_module(__name__, __file__)
