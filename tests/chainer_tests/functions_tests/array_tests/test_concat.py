import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import testing
from chainer.testing import backend


@testing.parameterize(*testing.product_dict(
    [
        {'shape': (2, 7, 3), 'axis': 1,
         'slices': [[slice(None), slice(None, 2)], [slice(None), slice(2, 5)],
                    [slice(None), slice(5, None)]]},
        {'shape': (7, 3), 'axis': 0,
         'slices': [slice(None, 2), slice(2, 5), slice(5, None)]},
        {'shape': (2,), 'axis': 0, 'slices': [slice(None, 1), slice(1, None)]},
        {'shape': (2,), 'axis': 0, 'slices': [()]},
        {'shape': (2, 7, 3), 'axis': 1,
         'slices': [[slice(None), slice(None, 2)], [slice(None), slice(2, 5)],
                    [slice(None), slice(5, None)]]},
        {'shape': (2, 7, 3), 'axis': 1,
         'slices': [[slice(None), slice(None, 2)], [slice(None), slice(2, 5)],
                    [slice(None), slice(5, None)]]},
        {'shape': (2, 7, 3), 'axis': -2,
         'slices': [[slice(None), slice(None, 2)], [slice(None), slice(2, 5)],
                    [slice(None), slice(5, None)]]},
        {'shape': (7, 3, 2, 2), 'axis': 0,
         'slices': [slice(None, 2), slice(2, 5), slice(5, None)]},
        {'shape': (2, 7, 3, 5), 'axis': 1,
         'slices': [[slice(None), slice(None, 2), slice(None)],
                    [slice(None), slice(2, 5), slice(None)],
                    [slice(None), slice(5, None), slice(None)]]},
        {'shape': (2, 7, 3, 5), 'axis': -1,
         'slices': [[slice(None), slice(None), slice(None), slice(None, 2)],
                    [slice(None), slice(None), slice(None), slice(2, 3)],
                    [slice(None), slice(None), slice(None), slice(3, None)]]},
        {'shape': (2, 7, 3, 5), 'axis': -3,
         'slices': [[slice(None), slice(None, 2), slice(None)],
                    [slice(None), slice(2, 5), slice(None)],
                    [slice(None), slice(5, None), slice(None)]]},
    ],
    [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ],
))
@backend.inject_backend_tests(
    ['test_forward', 'test_backward'],
    # CPU tests
    testing.product({
        'use_cuda': [False],
        'use_ideep': ['never', 'always'],
    })
    # GPU tests
    + [{'use_cuda': True}])
class TestConcat(unittest.TestCase):

    def setUp(self):
        shape = self.shape
        dtype = self.dtype

        y = numpy.arange(numpy.prod(shape), dtype=dtype).reshape(shape)
        xs = [y[s] for s in self.slices]

        self.y_expected = y
        self.inputs = xs

    def check_forward(self, inputs, backend_config):
        if backend_config.use_cuda:
            inputs = cuda.to_gpu(inputs)

        with backend_config:
            y = functions.concat(inputs, axis=self.axis)

        assert y.data.dtype == self.dtype
        testing.assert_allclose(self.y_expected, y.data, atol=0, rtol=0)
        assert isinstance(y.data.shape, tuple)

    def test_forward(self, backend_config):
        self.check_forward(self.inputs, backend_config)

    def check_backward(self, inputs, backend_config):
        if backend_config.use_cuda:
            inputs = cuda.to_gpu(inputs)

        inputs = [chainer.Variable(x) for x in inputs]

        with backend_config:
            y = functions.concat(inputs, axis=self.axis)
            y.grad = y.data
            y.backward()

        for x in inputs:
            testing.assert_allclose(x.data, x.grad, atol=0, rtol=0)

    def test_backward(self, backend_config):
        self.check_backward(self.inputs, backend_config)


class TestConcatInvalidAxisType(unittest.TestCase):

    def test_invlaid_axis_type(self):
        with self.assertRaises(TypeError):
            functions.Concat('a')


testing.run_module(__name__, __file__)
