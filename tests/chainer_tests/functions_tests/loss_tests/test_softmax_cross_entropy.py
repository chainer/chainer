import unittest

import numpy
import six

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
import chainerx


class SoftmaxCrossEntropyTestBase(object):

    def setUp(self):
        self.shape, self.ignore_index = self.shape_ignore
        if self.shape is None:
            if self.dtype == numpy.float16:
                self.x = numpy.array([[-5, 1]], dtype=self.dtype)
            else:
                self.x = numpy.array([[-1000, 1]], dtype=self.dtype)
            self.t = numpy.array([0], dtype=self.label_dtype)
        else:
            self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
            out_shape = (self.shape[0],) + self.shape[2:]
            self.t = numpy.random.randint(
                0, self.shape[1], out_shape).astype(self.label_dtype)
            if (self.ignore_index is not None and
                    len(self.ignore_index) <= self.t.ndim):
                self.t[self.ignore_index] = -1
        if self.reduce == 'mean':
            self.gy = numpy.random.uniform(-1, 1, ()).astype(self.x.dtype)
        else:
            self.gy = numpy.random.uniform(
                -1, 1, self.t.shape).astype(self.dtype)
        self.ggx = numpy.random.uniform(
            -1, 1, self.x.shape).astype(self.x.dtype)

        if self.weight_apply:
            self.class_weight = numpy.random.uniform(
                0, 10, (self.x.shape[1],)).astype(self.dtype)
        else:
            self.class_weight = None

        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_backward_options = {'atol': 5e-3, 'rtol': 5e-2}
            self.check_double_backward_options = {'atol': 5e-3, 'rtol': 5e-2}
        else:
            self.check_forward_options = {}
            self.check_backward_options = {}
            self.check_double_backward_options = {}

    def check_forward(self, x_data, t_data, class_weight, use_cudnn='always'):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data, requires_grad=False)
        with chainer.using_config('use_cudnn', use_cudnn):
            loss = functions.softmax_cross_entropy(
                x, t, normalize=self.normalize, reduce=self.reduce,
                cache_score=self.cache_score, class_weight=class_weight,
                enable_double_backprop=self.enable_double_backprop)
        self.assertEqual(loss.data.shape, self.gy.shape)
        self.assertEqual(loss.data.dtype, self.dtype)
        if (not self.enable_double_backprop
                and (chainer.backend.get_array_module(x_data)
                     is not chainerx)):
            assert (loss.creator.y is not None) == self.cache_score

        if self.reduce == 'mean':
            self.check_forward_with_reduce(
                float(loss.data), t_data, class_weight)
        else:
            self.check_forward_without_reduce(loss.data, t_data, class_weight)

    def expected_forward_with_reduce(self, x_data, t_data, class_weight):
        # Compute expected value
        loss_expect = 0.0
        count = 0
        x = numpy.rollaxis(x_data, 1, x_data.ndim).reshape(
            (t_data.size, x_data.shape[1]))
        t = t_data.ravel()
        for xi, ti in six.moves.zip(x, t):
            if ti == -1:
                continue
            log_z = numpy.ufunc.reduce(numpy.logaddexp, xi)
            if class_weight is None:
                loss_expect -= (xi - log_z)[ti]
            else:
                loss_expect -= (xi - log_z)[ti] * class_weight[ti]
            count += 1

        if self.normalize:
            if count == 0:
                loss_expect = 0.0
            else:
                loss_expect /= count
        else:
            if len(t_data) == 0:
                loss_expect = 0.0
            else:
                loss_expect /= len(t_data)
        return loss_expect

    def check_forward_with_reduce(self, loss_value, t_data, class_weight):
        loss_expect = self.expected_forward_with_reduce(
            self.x, self.t, backend.CpuDevice().send(class_weight))

        testing.assert_allclose(
            loss_expect, loss_value, **self.check_forward_options)

    def expected_forward_without_reduce(self, x_data, t_data, class_weight):
        x = numpy.rollaxis(x_data, 1, x_data.ndim).reshape(
            (t_data.size, x_data.shape[1]))
        t = t_data.ravel()

        loss_shape = x_data.shape[0:1] + x_data.shape[2:]
        loss_expect = numpy.zeros(loss_shape, x_data.dtype)
        for i, (ti, loss_idx) in enumerate(zip(t, numpy.ndindex(*loss_shape))):
            xi = x[i]
            if ti == -1:
                continue
            log_z = numpy.ufunc.reduce(numpy.logaddexp, xi)
            if class_weight is None:
                loss_expect[loss_idx] = -(xi - log_z)[ti]
            else:
                loss_expect[loss_idx] = -(xi - log_z)[ti] * class_weight[ti]

        return loss_expect

    def check_forward_without_reduce(self, loss_value, t_data, class_weight):
        loss_expect = self.expected_forward_without_reduce(
            self.x, self.t, backend.CpuDevice().send(class_weight))

        testing.assert_allclose(
            loss_expect, loss_value, **self.check_forward_options)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.t, self.class_weight)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.t),
            None if not self.weight_apply else cuda.to_gpu(self.class_weight))

    @attr.gpu
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.t),
            None if not self.weight_apply else cuda.to_gpu(self.class_weight),
            'never')

    @attr.chainerx
    def test_forward_chainerx_native(self):
        def conv(x):
            return chainer.backend.to_chainerx(x)

        self.check_forward(
            conv(self.x), conv(self.t),
            None if not self.weight_apply else conv(self.class_weight))

    @attr.chainerx
    @attr.gpu
    def test_forward_chainerx_cuda(self):
        def conv(x):
            return chainer.backend.to_chainerx(cuda.to_gpu(x))

        self.check_forward(
            conv(self.x), conv(self.t),
            None if not self.weight_apply else conv(self.class_weight))

    def check_backward(self, x_data, t_data, g_data, class_weight,
                       use_cudnn='always'):
        with chainer.using_config('use_cudnn', use_cudnn):
            def f(x):
                ys = functions.softmax_cross_entropy(
                    x, t_data, cache_score=self.cache_score,
                    class_weight=class_weight, reduce=self.reduce)
                return ys

            gradient_check.check_backward(
                f, x_data, g_data, dtype=numpy.float64,
                **self.check_backward_options)

    def test_backward_cpu(self):
        g_data = None
        if self.reduce == 'no':
            g_data = self.gy
        self.check_backward(self.x, self.t, g_data, self.class_weight)

    @attr.gpu
    def test_backward_gpu(self):
        g_data = None
        if self.reduce == 'no':
            g_data = cuda.to_gpu(self.gy)
        weight = None
        if not self.weight_apply:
            weight = cuda.to_gpu(self.class_weight)
        self.check_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.t), g_data, weight)

    @attr.gpu
    def test_backward_gpu_no_cudnn(self):
        g_data = None
        if self.reduce == 'no':
            g_data = cuda.to_gpu(self.gy)
        weight = None
        if not self.weight_apply:
            weight = cuda.to_gpu(self.class_weight)
        self.check_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.t), g_data, weight, 'never')

    @attr.chainerx
    def test_backward_chainerx_native(self):
        def conv(x):
            return chainer.backend.to_chainerx(x)

        g_data = None
        if self.reduce == 'no':
            g_data = conv(self.gy)
        weight = None
        if not self.weight_apply:
            weight = conv(self.class_weight)
        self.check_backward(
            conv(self.x), conv(self.t), g_data, weight)

    @attr.chainerx
    @attr.gpu
    def test_backward_chainerx_cuda(self):
        def conv(x):
            return chainer.backend.to_chainerx(cuda.to_gpu(x))

        g_data = None
        if self.reduce == 'no':
            g_data = conv(self.gy)
        weight = None
        if not self.weight_apply:
            weight = conv(self.class_weight)
        self.check_backward(
            conv(self.x), conv(self.t), g_data, weight)


test_cases = testing.product({
    # test each option flags
    'reduce': ['mean', 'no'],
    'cache_score': [True, False],
    'normalize': [True, False],
    'weight_apply': [True, False],
    'shape_ignore': [(None, None),
                     ((2, 3, 2, 2), (0, 1, 0))],
    'dtype': [numpy.float32],
    'label_dtype': [numpy.int32],
}) + testing.product({
    # test floating dtypes
    'reduce': ['mean', 'no'],
    'cache_score': [False],
    'normalize': [True],
    'weight_apply': [True],
    'shape_ignore': [(None, None),
                     ((2, 3), (slice(None),)),
                     ((2, 3, 2), (0,)),
                     ((2, 3, 2, 2), (0, 1, 0))],
    'dtype': [numpy.float16, numpy.float64],
    'label_dtype': [numpy.int32],
}) + testing.product({
    # test label dtypes
    'reduce': ['mean', 'no'],
    'cache_score': [False],
    'normalize': [True],
    'weight_apply': [True],
    'shape_ignore': [(None, None),
                     ((2, 3), (slice(None),)),
                     ((2, 3, 2), (0,)),
                     ((2, 3, 2, 2), (0, 1, 0))],
    'dtype': [numpy.float32],
    'label_dtype': [numpy.int8, numpy.int16, numpy.int64],
})


@testing.parameterize(*test_cases)
@testing.fix_random()
class TestSoftmaxCrossEntropyDisableDoubleBackprop(
        SoftmaxCrossEntropyTestBase, unittest.TestCase):

    enable_double_backprop = False


@testing.parameterize(*test_cases)
@testing.fix_random()
class TestSoftmaxCrossEntropyEnableDoubleBackprop(
        SoftmaxCrossEntropyTestBase, unittest.TestCase):

    enable_double_backprop = True

    def check_double_backward(self, x_data, t_data, gy_data, ggx_data,
                              class_weight, use_cudnn='always'):
        def f(x):
            return functions.softmax_cross_entropy(
                x, t_data, self.normalize, self.cache_score, class_weight,
                reduce=self.reduce, enable_double_backprop=True)

        with chainer.using_config('use_cudnn', use_cudnn):
            gradient_check.check_double_backward(
                f, x_data, gy_data, ggx_data, dtype=numpy.float64,
                **self.check_double_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(
            self.x, self.t, self.gy, self.ggx, self.class_weight)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.t),
            cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx),
            None if not self.weight_apply else cuda.to_gpu(self.class_weight))

    @attr.gpu
    def test_double_backward_gpu_no_cudnn(self):
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.t),
            cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx),
            None if not self.weight_apply else cuda.to_gpu(self.class_weight),
            'never')

    @attr.chainerx
    def test_double_backward_chainerx_native(self):
        def conv(x):
            return chainer.backend.to_chainerx(x)

        args = [conv(a) for a in [self.x, self.t, self.gy, self.ggx]]
        args.append(None if not self.weight_apply else conv(self.class_weight))
        self.check_double_backward(*args)

    @attr.chainerx
    @attr.gpu
    def test_double_backward_chainerx_cuda(self):
        def conv(x):
            return chainer.backend.to_chainerx(cuda.to_gpu(x))

        args = [conv(a) for a in [self.x, self.t, self.gy, self.ggx]]
        args.append(None if not self.weight_apply else conv(self.class_weight))
        self.check_double_backward(*args)


@testing.parameterize(*testing.product_dict(
    [
        {'t_value': -2, 'valid': False},
        {'t_value': 3, 'valid': False},
        {'t_value': -1, 'valid': True}  # -1 is ignore_label
    ],
    [
        {'enable_double_backprop': True},
        {'enable_double_backprop': False}
    ]
))
class TestSoftmaxCrossEntropyValueCheck(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 2)).astype(numpy.float32)
        # `0` is required to avoid NaN
        self.t = numpy.array([self.t_value, 0], dtype=numpy.int32)
        self.original_debug = chainer.is_debug()
        chainer.set_debug(True)

    def tearDown(self):
        chainer.set_debug(self.original_debug)

    def check_value_check(self, x_data, t_data, use_cudnn):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)

        with chainer.using_config('use_cudnn', use_cudnn):
            if self.valid:
                # Check if it throws nothing
                functions.softmax_cross_entropy(
                    x, t, enable_double_backprop=self.enable_double_backprop)
            else:
                with self.assertRaises(ValueError):
                    functions.softmax_cross_entropy(
                        x, t,
                        enable_double_backprop=self.enable_double_backprop)

    def test_value_check_cpu(self):
        self.check_value_check(self.x, self.t, 'never')

    @attr.gpu
    def test_value_check_gpu(self):
        self.check_value_check(self.x, self.t, 'never')

    @attr.gpu
    def test_value_check_gpu_cudnn(self):
        self.check_value_check(cuda.to_gpu(self.x), cuda.to_gpu(self.t),
                               'always')


@testing.parameterize(*testing.product({
    'use_cudnn': ['always', 'auto', 'never'],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@attr.cudnn
class TestSoftmaxCrossEntropyCudnnCall(unittest.TestCase):

    def setUp(self):
        self.x = cuda.cupy.random.uniform(-1, 1, (4, 3)).astype(self.dtype)
        self.t = cuda.cupy.random.randint(0, 3, (4,)).astype(numpy.int32)

    def forward(self):
        x = chainer.Variable(self.x)
        t = chainer.Variable(self.t)
        return functions.softmax_cross_entropy(
            x, t, enable_double_backprop=False)

    def test_call_cudnn_forward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            with testing.patch('cupy.cudnn.softmax_forward') as func:
                self.forward()
                self.assertEqual(func.called,
                                 chainer.should_use_cudnn('>=auto'))


# Note that SoftmaxCrossEntropy does not use cudnn on backward
@testing.parameterize(
    {'enable_double_backprop': True},
    {'enable_double_backprop': False},
)
class TestClassWeightAssertion(unittest.TestCase):

    def setUp(self):
        self.x = numpy.array([[0, 1], [2, 3]])
        self.t = numpy.array([0, 1])

    def test_ndim_assertion(self):
        wrong_ndim_class_weight = numpy.array([[0, 0]], dtype='f')
        with self.assertRaises(ValueError):
            functions.softmax_cross_entropy(
                self.x, self.t, class_weight=wrong_ndim_class_weight,
                enable_double_backprop=self.enable_double_backprop)

    def test_dtype_assertion(self):
        wrong_dtype_class_weight = numpy.array([0, 0], dtype=numpy.int32)
        with self.assertRaises(ValueError):
            functions.softmax_cross_entropy(
                self.x, self.t, class_weight=wrong_dtype_class_weight,
                enable_double_backprop=self.enable_double_backprop)

    def test_variable_assertion(self):
        wrong_inst_class_weight = chainer.Variable(
            numpy.array([0, 0], dtype='f'))
        with self.assertRaises(ValueError):
            functions.softmax_cross_entropy(
                self.x, self.t, class_weight=wrong_inst_class_weight,
                enable_double_backprop=self.enable_double_backprop)


@testing.parameterize(*testing.product({
    'enable_double_backprop': [True, False],
}))
class TestSoftmaxCrossEntropyInvalidReduce(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3)).astype('f')
        self.t = numpy.zeros((2,), 'i')

    def check_invalid_reduce(self, x, t):
        with self.assertRaises(ValueError):
            functions.softmax_cross_entropy(
                x, t,
                reduce='unknown_reduce_type',
                enable_double_backprop=self.enable_double_backprop)

    def test_invalid_reduce_cpu(self):
        self.check_invalid_reduce(self.x, self.t)

    @attr.gpu
    def test_invalid_reduce_gpu(self):
        self.check_invalid_reduce(cuda.to_gpu(self.x), cuda.to_gpu(self.t))


@testing.parameterize(*testing.product({
    'ignore_label': [-2, 9],
    'reduce': ['mean', 'no'],
    'enable_double_backprop': [False, True],
    'class_weight': [None, numpy.ones((3,), dtype=numpy.float32)]})
)
class TestNonDefaultIgnoreLabel(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)
        self.t = numpy.full((2,), self.ignore_label, dtype=numpy.int32)
        if self.reduce == 'mean':
            gy_shape = ()
        else:
            gy_shape = (2,)
        self.gy = numpy.random.uniform(-1, 1, gy_shape).astype(numpy.float32)
        self.ggx = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)

    def check_forward(self, xp):
        x = xp.asarray(self.x)
        t = xp.asarray(self.t)
        if self.class_weight is not None:
            class_weight = xp.asarray(self.class_weight)
        else:
            class_weight = None
        loss = functions.softmax_cross_entropy(
            x, t, reduce=self.reduce,
            class_weight=class_weight,
            ignore_label=self.ignore_label,
            enable_double_backprop=self.enable_double_backprop)
        if self.reduce == 'mean':
            expect = 0.
        else:
            expect = numpy.zeros((2,), dtype=numpy.float32)
        testing.assert_allclose(loss.data, expect)

    def test_forward_cpu(self):
        self.check_forward(numpy)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.cupy)

    def check_backward(self, xp):
        x = xp.asarray(self.x)
        t = xp.asarray(self.t)
        gy = xp.asarray(self.gy)
        if self.class_weight is not None:
            class_weight = xp.asarray(self.class_weight)
        else:
            class_weight = None

        def f(x_, t_):
            return functions.softmax_cross_entropy(
                x_, t_, class_weight=class_weight, reduce=self.reduce,
                ignore_label=self.ignore_label,
                enable_double_backprop=self.enable_double_backprop)

        gradient_check.check_backward(f, (x, t), gy)

    def test_backward_cpu(self):
        self.check_backward(numpy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.cupy)

    def check_double_backward(self, xp):
        x = xp.asarray(self.x)
        t = xp.asarray(self.t)
        gy = xp.asarray(self.gy)
        ggx = xp.asarray(self.ggx)
        if self.class_weight is not None:
            class_weight = xp.asarray(self.class_weight)
        else:
            class_weight = None

        def f(x_):
            return functions.softmax_cross_entropy(
                x_, t, class_weight=class_weight, reduce=self.reduce,
                ignore_label=self.ignore_label,
                enable_double_backprop=True)

        gradient_check.check_double_backward(f, x, gy, ggx)

    def test_double_backward_cpu(self):
        self.check_double_backward(numpy)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(cuda.cupy)


@testing.parameterize(*(testing.product({
    'shape_ignore': [(None, None),
                     ((2, 3), (slice(None),)),
                     ((2, 3, 2), (0,)),
                     ((2, 3, 2, 2), (0, 1, 0))],
    'normalize': [True, False],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'weight_apply': [False, True],
})))
class TestForwardConsistency(unittest.TestCase):

    # This test case checks if forward propagation of
    # double backpropable impl. and non-double backpropable impl.
    # agree.

    def setUp(self):
        self.shape, self.ignore_index = self.shape_ignore
        if self.shape is None:
            if self.dtype == numpy.float16:
                self.x = numpy.array([[-5, 1]], dtype=self.dtype)
            else:
                self.x = numpy.array([[-1000, 1]], dtype=self.dtype)
            self.t = numpy.array([0], dtype=numpy.int32)
        else:
            self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
            out_shape = (self.shape[0],) + self.shape[2:]
            self.t = numpy.random.randint(
                0, self.shape[1], out_shape).astype(numpy.int32)
            if (self.ignore_index is not None and
                    len(self.ignore_index) <= self.t.ndim):
                self.t[self.ignore_index] = -1
        if self.weight_apply:
            self.class_weight = numpy.random.uniform(
                0, 10, (self.x.shape[1],)).astype(self.dtype)
        else:
            self.class_weight = None

    def check_consistency(self, xp):

        if self.class_weight is None:
            class_weight = None
        else:
            class_weight = xp.asarray(self.class_weight)

        x = xp.asarray(self.x)
        t = xp.asarray(self.t)

        def f(enable_double_backprop):
            kwargs = {
                'normalize': self.normalize,
                'class_weight': class_weight,
                'enable_double_backprop': enable_double_backprop
            }

            return functions.softmax_cross_entropy(x, t, **kwargs).data

        loss_single = f(False)
        loss_double = f(True)

        check_forward_options = {}
        if self.dtype == numpy.float16:
            check_forward_options = {'atol': 5e-4, 'rtol': 5e-3}
        testing.assert_allclose(
            loss_single, loss_double, **check_forward_options)

    def test_consistency_cpu(self):
        self.check_consistency(numpy)

    @attr.gpu
    def test_consistency_gpu_always(self):
        with chainer.using_config('use_cudnn', 'always'):
            self.check_consistency(cuda.cupy)

    @attr.gpu
    def test_consistency_gpu_auto(self):
        with chainer.using_config('use_cudnn', 'auto'):
            self.check_consistency(cuda.cupy)

    @attr.gpu
    def test_consistency_gpu_never(self):
        with chainer.using_config('use_cudnn', 'never'):
            self.check_consistency(cuda.cupy)


testing.run_module(__name__, __file__)
