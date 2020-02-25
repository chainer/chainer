import numpy
import six

from chainer import functions
from chainer import testing
from chainer.testing import backend


def _sigmoid(x):
    half = x.dtype.type(0.5)
    return numpy.tanh(x * half) * half + half


def _shaped_random_array(shape, dtype):
    return numpy.random.uniform(-1, 1, shape).astype(dtype)


@testing.parameterize(*testing.product_dict(
    [
        {'c_dim': 2, 'num_c': 1, 'batch_size': 3},
        {'c_dim': 0, 'num_c': 4, 'batch_size': 5},
        {'c_dim': 12, 'num_c': 1, 'batch_size': 0},
    ], [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ]
))
@testing.fix_random()
@backend.inject_backend_tests(
    None,
    # ChainerX tests
    testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0'],
    })
    # CPU tests
    + testing.product({
        'use_cuda': [False],
        'use_ideep': ['never', 'always'],
    })
    # GPU tests
    + testing.product([
        [{'use_cuda': True}],

        # Without cuDNN
        testing.product({
            'use_cudnn': ['never'],
        })
        # With cuDNN
        + testing.product({
            'use_cudnn': ['always'],
            'cudnn_deterministic': [True, False],
            'autotune': [True, False],
        })]))
class TestTreeLSTM(testing.FunctionTestCase):

    def setUp(self):
        if self.dtype == numpy.float16:
            self.check_forward_options.update({
                'rtol': 1e-2, 'atol': 1e-2})
            self.check_backward_options.update({
                'rtol': 1e-2, 'atol': 1e-2})
            self.check_double_backward_options.update({
                'rtol': 5e-3, 'atol': 5e-2})
        self.skip_double_backward_test = True

    def generate_inputs(self):
        c_dim = self.c_dim
        num_c = self.num_c
        batch_size = self.batch_size
        c_shape = (batch_size, c_dim)
        input_shape = (batch_size, c_dim * (num_c + 3))
        inputs = []
        for i in range(num_c):
            inputs.append(_shaped_random_array(c_shape, self.dtype))
        inputs.append(_shaped_random_array(input_shape, self.dtype))
        return tuple(inputs)

    def forward(self, inputs, device):
        out = functions.tree_lstm(*list(inputs))
        return out

    def forward_expected(self, inputs):
        def _extract_gates(x, n_split=5):
            r = x.reshape(
                (x.shape[0], n_split, x.shape[1] // n_split) + x.shape[2:])
            return [r[:, i, :] for i in six.moves.range(n_split)]

        cs, x = inputs[:-1], inputs[-1]
        n_ary = len(cs)
        gates = _extract_gates(x, 3 + n_ary)
        a, i, o = gates[:3]
        fs = gates[3:]
        a = numpy.tanh(a)
        i = _sigmoid(i)
        o = _sigmoid(o)
        fs = [_sigmoid(f) for f in fs]
        c = a * i + sum(f * c for f, c in zip(fs, cs))
        h = o * numpy.tanh(c)
        return c, h


testing.run_module(__name__, __file__)
