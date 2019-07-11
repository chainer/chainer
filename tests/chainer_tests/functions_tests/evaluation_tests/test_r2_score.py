import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr
from chainer.utils import force_array
from chainer.utils import type_check


def r2_score(pred, true, sample_weight=None, multioutput='uniform_average'):
    SS_res = numpy.asarray(
        numpy.sum((pred - true) ** 2, axis=0))
    SS_tot = numpy.asarray(
        numpy.sum((true - numpy.mean(true, axis=0)) ** 2, axis=0))

    if multioutput == 'uniform_average':
        if numpy.any(SS_tot == 0):
            return 0.0
        else:
            return (1 - SS_res / SS_tot).mean()
    elif multioutput == 'raw_values':
        if numpy.any(SS_tot == 0):
            # Assign dummy value to avoid zero-division
            SS_tot_iszero = SS_tot == 0
            SS_tot[SS_tot_iszero] = 1

            return numpy.where(SS_tot_iszero, 0.0, 1 - SS_res / SS_tot)
        else:
            return 1 - SS_res / SS_tot


@testing.parameterize(
    *testing.product_dict(
        [{'x_shape': (10,), 't_shape': (10,)},
         {'x_shape': (10, 1), 't_shape': (10, 1)},
         {'x_shape': (10, 5), 't_shape': (10, 5)},
         {'x_shape': (10, 5, 4), 't_shape': (10, 5, 4)}],
        [{'t_input': 'random'}, {'t_input': 'zero'}],
        [{'multioutput': 'uniform_average'},
         {'multioutput': 'raw_values'}],
        [{'sample_weight': None}],
        [{'dtype': numpy.float16},
         {'dtype': numpy.float32},
         {'dtype': numpy.float64}]
    )
)
@testing.fix_random()
@testing.inject_backend_tests(
    None,
    # CPU tests
    [
        {},
    ]
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'cuda_device': [0, 1],
    })
    # ChainerX tests
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0', 'cuda:1'],
    })
)
class TestAccuracy(testing.FunctionTestCase):

    def setUp(self):
        self.skip_backward_test = True
        self.skip_double_backward_test = True

        if self.dtype == numpy.float16:
            self.check_forward_options.update({'atol': 1e-2, 'rtol': 1e-2})

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, self.x_shape).astype(self.dtype)
        if self.t_input == 'random':
            t = numpy.random.uniform(-1, 1, self.t_shape).astype(self.dtype)
        else:
            t = numpy.zeros(self.t_shape).astype(self.dtype)
        return x, t

    def forward(self, inputs, device):
        x, t = inputs
        y = functions.r2_score(x, t, self.sample_weight, self.multioutput)
        return y,

    def forward_expected(self, inputs):
        x, t = inputs
        expected = r2_score(x, t, sample_weight=self.sample_weight,
                            multioutput=self.multioutput)
        expected = force_array(expected, self.dtype)
        return expected,


@testing.parameterize(
    {'x_shape': (10, 3), 't_shape': (4,)},
    {'x_shape': (10, 3, 2), 't_shape': (10,)},
    {'x_shape': (10, 3, 1, 2), 't_shape': (10,)},
    {'x_shape': (10, 3, 4), 't_shape': (10, 5)},
    {'x_shape': (10, 3, 5, 2), 't_shape': (10, 5)},
    {'x_shape': (10, 3, 5, 1, 2), 't_shape': (10, 5)},
)
class TestInvalidShape(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1,
                                      self.x_shape).astype(numpy.float32)
        self.t = numpy.random.randint(3, size=self.t_shape).astype(numpy.int32)

    def check_invalid_shape(self, xp):
        x = chainer.Variable(xp.asarray(self.x))
        t = chainer.Variable(xp.asarray(self.t))
        with self.assertRaises(type_check.InvalidType):
            chainer.functions.accuracy(x, t)

    def test_invalid_shape_cpu(self):
        self.check_invalid_shape(numpy)

    @attr.gpu
    def test_invalid_shape_gpu(self):
        self.check_invalid_shape(cuda.cupy)


testing.run_module(__name__, __file__)
