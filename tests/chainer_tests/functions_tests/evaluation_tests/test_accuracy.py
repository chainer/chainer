import unittest

import numpy
import six

import chainer
from chainer.backends import cuda
from chainer import testing
from chainer import functions
from chainer.testing import attr
from chainer.utils import force_array
from chainer.utils import type_check


def accuracy(x, t, ignore_label):
    x_ = numpy.rollaxis(x, 1, x.ndim).reshape(t.size, -1)
    t_ = t.ravel()

    if ignore_label is not None:
        count = 0
        for i in six.moves.range(t_.size):
            pred = x_[i].argmax()
            if t_[i] != ignore_label and pred == t_[i]:
                count += 1
        total = (t_ != ignore_label).sum()
    else:
        count = 0
        for i in six.moves.range(t_.size):
            pred = x_[i].argmax()
            if pred == t_[i]:
                count += 1
        total = t_.size

    if total == 0:
        return 0.0
    else:
        return float(count) / total


@testing.parameterize(
    *testing.product_dict(
        [{'x_shape': (10, 3), 't_shape': (10,)},
         {'x_shape': (10, 3, 1), 't_shape': (10,)},
         {'x_shape': (10, 3, 1, 1), 't_shape': (10,)},
         {'x_shape': (10, 3, 5), 't_shape': (10, 5)},
         {'x_shape': (10, 3, 5, 4), 't_shape': (10, 5, 4)},
         {'x_shape': (10, 3, 5, 4, 1), 't_shape': (10, 5, 4)},
         {'x_shape': (10, 3, 5, 4, 1, 1), 't_shape': (10, 5, 4)}],
        [{'ignore_label': None, 't_data': 'randint'},
         {'ignore_label': 0, 't_data': 'randint'},
         {'ignore_label': 0, 't_data': 'zero'}],
        [{'dtype': numpy.float16},
         {'dtype': numpy.float32},
         {'dtype': numpy.float64}],
        [{'label_dtype': numpy.int8},
         {'label_dtype': numpy.int16},
         {'label_dtype': numpy.int32},
         {'label_dtype': numpy.int64}]
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
            self.check_forward_options.update({'atol': 1e-4, 'rtol': 1e-3})

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, self.x_shape).astype(self.dtype)
        if self.t_data == 'randint':
            t = numpy.random.randint(
                3, size=self.t_shape).astype(self.label_dtype)
        elif self.t_data == 'zero':
            t = numpy.zeros(self.t_shape).astype(self.label_dtype)
        return x, t

    def forward(self, inputs, device):
        x, t = inputs
        return functions.accuracy(x, t, self.ignore_label),

    def forward_expected(self, inputs):
        x, t = inputs
        expected = accuracy(x, t, self.ignore_label)
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
            functions.accuracy(x, t)

    def test_invalid_shape_cpu(self):
        self.check_invalid_shape(numpy)

    @attr.gpu
    def test_invalid_shape_gpu(self):
        self.check_invalid_shape(cuda.cupy)


testing.run_module(__name__, __file__)
