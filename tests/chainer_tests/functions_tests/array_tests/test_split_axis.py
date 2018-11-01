import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import testing
from chainer.testing import backend


def inject_backend_tests(method_names):
    decorator = backend.inject_backend_tests(
        method_names,
        # CPU tests
        testing.product({
            'use_cuda': [False],
            'use_ideep': ['never', 'always'],
        })
        # GPU tests
        + [{'use_cuda': True}]
        # ChainerX tests
        + [
            {'use_chainerx': True, 'chainerx_device': 'native:0'},
            {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
        ])
    return decorator


@testing.parameterize(*testing.product_dict(
    [
        {'shape': (2, 7, 3), 'axis': 1, 'ys_section': [2, 5],
         'slices': [
             (slice(None), slice(None, 2)),
             (slice(None), slice(2, 5)),
             (slice(None), slice(5, None))]},
        {'shape': (7, 3), 'axis': 0, 'ys_section': [2, 5],
         'slices': [slice(None, 2), slice(2, 5), slice(5, None)]},
        {'shape': (7, 0), 'axis': 0, 'ys_section': [2, 5],
         'slices': [slice(None, 2), slice(2, 5), slice(5, None)]},
        {'shape': (2, 9, 3), 'axis': 1, 'ys_section': 3,
         'slices': [
             (slice(None), slice(None, 3)),
             (slice(None), slice(3, 6)),
             (slice(None), slice(6, None))]},
        {'shape': (2, 6, 3), 'axis': 1, 'ys_section': 3,
         'slices': [
             (slice(None), slice(None, 2)),
             (slice(None), slice(2, 4)),
             (slice(None), slice(4, None))]},
        {'shape': (2,), 'axis': 0, 'ys_section': [1],
         'slices': [slice(None, 1), slice(1, None)]},
        {'shape': (2,), 'axis': 0, 'ys_section': [],
         'slices': [slice(None, None)]},
        {'shape': (2, 7, 3), 'axis': 1, 'ys_section': [2, 5],
         'slices': [
             (slice(None), slice(None, 2)),
             (slice(None), slice(2, 5)),
             (slice(None), slice(5, None))]},
        {'shape': (2, 7, 3), 'axis': 1, 'ys_section': [0],
         'slices': [
             (slice(None), slice(None, 0)),
             (slice(None), slice(0, 7))]
         },
        {'shape': (2, 7, 3), 'axis': 1, 'ys_section': [7],
         'slices': [
             (slice(None), slice(None, 7)),
             (slice(None), slice(7, 7))]
         },
        {'shape': (2, 7, 3, 2), 'axis': 1, 'ys_section': [2, 5],
         'slices': [
             (slice(None), slice(None, 2)),
             (slice(None), slice(2, 5)),
             (slice(None), slice(5, None))]},
        {'shape': (2, 7, 3, 2), 'axis': 1, 'ys_section': [0],
         'slices': [
             (slice(None), slice(None, 0)),
             (slice(None), slice(0, 7))]
         },
        {'shape': (10, 4, 3, 2), 'axis': 0, 'ys_section': 1,
         'slices': [slice(None, None)]
         },
        {'shape': (10, 4, 3, 2), 'axis': 0, 'ys_section': 2,
         'slices': [slice(None, 5), slice(5, None)]
         },
        {'shape': (10, 4, 3, 2), 'axis': 0, 'ys_section': [],
         'slices': [slice(None, None)]
         },
        {'shape': (10, 4, 3, 2), 'axis': 0, 'ys_section': [0, 5],
         'slices': [slice(0, 0), slice(0, 5), slice(5, None)]
         },
        {'shape': (10, 4, 3, 2), 'axis': 0, 'ys_section': [0, 0, 5],
         'slices': [slice(0, 0), slice(0, 0), slice(None, 5), slice(5, None)]
         },
        {'shape': (10, 4, 3, 2), 'axis': 0, 'ys_section': [2, 3, 5],
         'slices': [slice(None, 2), slice(2, 3), slice(3, 5), slice(5, None)]
         },
        {'shape': (10, 4, 3, 2), 'axis': 0,
         'ys_section': numpy.asarray([2, 3, 5]),
         'slices': [slice(None, 2), slice(2, 3), slice(3, 5), slice(5, None)]
         },
        {'shape': (10, 4, 3, 2), 'axis': 0, 'ys_section': [2, 3, 3, 5],
         'slices': [slice(None, 2), slice(2, 3), slice(3, 3), slice(3, 5),
                    slice(5, None)]
         },
        {'shape': (5, 5, 3, 8), 'axis': 3, 'ys_section': 2,
         'slices': [
             (slice(None, None), slice(None, None), slice(None, None),
              slice(None, 4)),
             (slice(None, None), slice(None, None), slice(None, None),
              slice(4, None))]
         },
        {'shape': (5, 8, 3, 2), 'axis': -3, 'ys_section': 2,
         'slices': [(slice(None, None), slice(None, 4)),
                    (slice(None, None), slice(4, None))]
         },
        {'shape': (5, 8, 3, 2), 'axis': 1, 'ys_section': 2,
         'slices': [(slice(None, None), slice(None, 4)),
                    (slice(None, None), slice(4, None))]
         },
        {'shape': (5, 4, 3, 4), 'axis': -1, 'ys_section': 2,
         'slices': [
             (slice(None, None), slice(None, None), slice(None, None),
              slice(None, 2)),
             (slice(None, None), slice(None, None), slice(None, None),
              slice(2, None))]
         },
        {'shape': (10, 4, 3, 2), 'axis': 0, 'ys_section': numpy.array([]),
         'slices': [slice(None, None)]
         },
    ],
    [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ],
))
@inject_backend_tests(['test_forward', 'test_backward'])
class TestSplitAxis(unittest.TestCase):

    def setUp(self):
        shape = self.shape
        dtype = self.dtype

        x = numpy.arange(numpy.prod(shape), dtype=dtype).reshape(shape)
        self.ys_expected = [x[s] for s in self.slices]
        self.inputs = [x]

    def check_forward(self, inputs, backend_config):
        inputs = backend_config.get_array(inputs)

        x, = inputs
        x = chainer.Variable(x)

        with backend_config:
            ys = functions.split_axis(
                x, self.ys_section, self.axis, force_tuple=True)

        for yd, y in zip(self.ys_expected, ys):
            assert y.data.dtype == self.dtype
            assert isinstance(y.data.shape, tuple)
            testing.assert_allclose(yd, y.data, atol=0, rtol=0)

    def test_forward(self, backend_config):
        # TODO(hvy): Support it
        if self.dtype == numpy.float16 and backend_config.use_chainerx:
            raise unittest.SkipTest('ChainerX does not support float16')
        self.check_forward(self.inputs, backend_config)

    def check_backward(self, inputs, backend_config):
        inputs = backend_config.get_array(inputs)

        x, = inputs
        x = chainer.Variable(x)

        with backend_config:
            ys = functions.split_axis(
                x, self.ys_section, self.axis, force_tuple=True)
            for y in ys:
                y.grad = y.data
            ys[0].backward()

        testing.assert_allclose(x.data, x.grad, atol=0, rtol=0)

    def test_backward(self, backend_config):
        # TODO(hvy): Support it
        if self.dtype == numpy.float16 and backend_config.use_chainerx:
            raise unittest.SkipTest('ChainerX does not support float16')
        self.check_backward(self.inputs, backend_config)


@inject_backend_tests(['test_backward'])
class TestSplitAxisNone(unittest.TestCase):

    def setUp(self):
        self.ys_section = [1]
        self.axis = 0

        self.inputs = [numpy.array([1, 2], dtype=numpy.float32)]

    def check_backward(self, inputs, backend_config):
        if backend_config.use_cuda:
            inputs = cuda.to_gpu(inputs)

        x, = inputs
        x = chainer.Variable(x)

        with backend_config:
            ys = functions.split_axis(x, self.ys_section, self.axis)
            # Only set ys[0]
            ys[0].grad = ys[0].data
            ys[0].backward()

        gx = numpy.array([1, 0])
        testing.assert_allclose(gx, x.grad, atol=0, rtol=0)

    def test_backward(self, backend_config):
        self.check_backward(self.inputs, backend_config)


@inject_backend_tests(['test_forward_force_tuple', 'test_forward_single'])
class TestSplitAxisForceArray(unittest.TestCase):

    def setUp(self):
        self.axis = 1
        self.inputs = [numpy.arange(42, dtype=numpy.float32).reshape(2, 7, 3)]

    def check_forward_force_tuple(self, inputs, backend_config):
        if backend_config.use_cuda:
            inputs = cuda.to_gpu(inputs)

        x, = self.inputs
        x = chainer.Variable(x)

        with backend_config:
            ys = functions.split_axis(x, 1, self.axis, force_tuple=True)

        assert isinstance(ys, tuple)
        assert len(ys) == 1

    def test_forward_force_tuple(self, backend_config):
        self.check_forward_force_tuple(self.inputs, backend_config)

    def check_forward_single(self, inputs, backend_config):
        if backend_config.use_cuda:
            inputs = cuda.to_gpu(inputs)

        x, = self.inputs
        x = chainer.Variable(x)

        with backend_config:
            ys = functions.split_axis(x, 1, self.axis, force_tuple=False)

        assert isinstance(ys, chainer.Variable)

    def test_forward_single(self, backend_config):
        self.check_forward_single(self.inputs, backend_config)


class TestSplitAxisInvalidSections(unittest.TestCase):

    def setUp(self):
        self.default_debug = chainer.is_debug()
        chainer.set_debug(True)

    def tearDown(self):
        chainer.set_debug(self.default_debug)

    def test_invalid_sections(self):
        x = numpy.zeros((2, 3, 4), dtype='f')
        with self.assertRaises(ValueError):
            functions.split_axis(x, [2, 1], 1)


testing.run_module(__name__, __file__)
