import unittest

import mock
import numpy
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(*(testing.product({
    'shape': [None, (2, 3), (2, 3, 2), (2, 3, 2, 2)],
    'cache_score': [True, False],
    'normalize': [True, False],
    'ignore_index': [None, (slice(None),), (0,), (0, 1), (0, 1, 0)],
    'dtype': [numpy.float32],
    'weight_apply': [False, True],
}) + testing.product({
    'shape': [None, (2, 3), (2, 3, 2), (2, 3, 2, 2)],
    'cache_score': [False],
    'normalize': [True],
    'ignore_index': [(0, 1)],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'weight_apply': [False, True],
})))
class TestSoftmaxCrossEntropy(unittest.TestCase):

    def setUp(self):
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
        self.check_forward_options = {}
        self.check_backward_options = {'dtype': numpy.float64}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 5e-4, 'rtol': 5e-3}
        if self.weight_apply:
            self.class_weight = numpy.random.uniform(
                0, 10, (self.x.shape[1],)).astype(self.dtype)
        else:
            self.class_weight = None

    def check_forward(self, x_data, t_data, class_weight, use_cudnn='always'):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        with chainer.using_config('use_cudnn', use_cudnn):
            loss = functions.softmax_cross_entropy(
                x, t, normalize=self.normalize,
                cache_score=self.cache_score, class_weight=class_weight)
        self.assertEqual(loss.data.shape, ())
        self.assertEqual(loss.data.dtype, self.dtype)
        self.assertEqual(hasattr(loss.creator, 'y'), self.cache_score)
        loss_value = float(cuda.to_cpu(loss.data))

        # Compute expected value
        loss_expect = 0.0
        count = 0
        x = numpy.rollaxis(self.x, 1, self.x.ndim).reshape(
            (self.t.size, self.x.shape[1]))
        t = self.t.ravel()
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
            loss_expect /= len(t_data)

        testing.assert_allclose(
            loss_expect, loss_value, **self.check_forward_options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x, self.t, self.class_weight)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.t),
            None if not self.weight_apply else cuda.to_gpu(self.class_weight))

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.t),
            None if not self.weight_apply else cuda.to_gpu(self.class_weight),
            'never')

    def check_backward(self, x_data, t_data, class_weight, use_cudnn='always'):
        with chainer.using_config('use_cudnn', use_cudnn):
            func = functions.SoftmaxCrossEntropy(
                cache_score=self.cache_score, class_weight=class_weight)
            gradient_check.check_backward(
                func, (x_data, t_data), None, eps=0.02,
                **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.t, self.class_weight)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.t),
            None if not self.weight_apply else cuda.to_gpu(self.class_weight))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.t),
            None if not self.weight_apply else cuda.to_gpu(self.class_weight),
            'never')


@testing.parameterize(
    {'t_value': -2, 'valid': False},
    {'t_value': 3, 'valid': False},
    {'t_value': -1, 'valid': True},  # -1 is ignore_label
)
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
                functions.softmax_cross_entropy(x, t)
            else:
                with self.assertRaises(ValueError):
                    functions.softmax_cross_entropy(x, t)

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
        return functions.softmax_cross_entropy(x, t)

    @unittest.skipIf(cuda.cudnn_enabled and
                     cuda.cudnn.cudnn.getVersion() < 3000,
                     'Only cudnn ver>=3 supports softmax-log')
    def test_call_cudnn_forward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            with mock.patch('cupy.cudnn.cudnn.softmaxForward') as func:
                self.forward()
                self.assertEqual(func.called,
                                 chainer.should_use_cudnn('>=auto'))

    # Note that SoftmaxCrossEntropy does not use cudnn on backward


class TestClassWeightAssertion(unittest.TestCase):

    def setUp(self):
        self.x = numpy.array([[0, 1], [2, 3]])
        self.t = numpy.array([0, 1])

    def test_ndim_assertion(self):
        wrong_ndim_class_weight = numpy.array([[0, 0]], dtype='f')
        with self.assertRaises(ValueError):
            functions.softmax_cross_entropy(
                self.x, self.t, class_weight=wrong_ndim_class_weight)

    def test_dtype_assertion(self):
        wrong_dtype_class_weight = numpy.array([0, 0], dtype=numpy.int32)
        with self.assertRaises(ValueError):
            functions.softmax_cross_entropy(
                self.x, self.t, class_weight=wrong_dtype_class_weight)

    def test_variable_assertion(self):
        wrong_inst_class_weight = chainer.Variable(
            numpy.array([0, 0], dtype='f'))
        with self.assertRaises(ValueError):
            functions.softmax_cross_entropy(
                self.x, self.t, class_weight=wrong_inst_class_weight)


@testing.parameterize(*(testing.product({
    'shape': [None, (2, 3), (2, 3, 2), (2, 3, 2, 2)],
    'cache_score': [True, False],
    'ignore_index': [None, (slice(None),), (0,), (0, 1), (0, 1, 0)],
    'dtype': [numpy.float32],
    'weight_apply': [False, True],
    'use_cudnn': ['always', 'auto', 'never'],
}) + testing.product({
    'shape': [None, (2, 3), (2, 3, 2), (2, 3, 2, 2)],
    'cache_score': [False],
    'ignore_index': [(0, 1)],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'weight_apply': [False, True],
    'use_cudnn': ['always', 'auto', 'never'],
})))
class TestElementwiseSoftmaxCrossEntropy(unittest.TestCase):

    def setUp(self):
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
        self.g = numpy.random.uniform(-1, 1, self.t.shape).astype(self.dtype)
        self.check_forward_options = {}
        self.check_backward_options = {'dtype': numpy.float64}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 5e-4, 'rtol': 5e-3}
        if self.weight_apply:
            self.class_weight = numpy.random.uniform(
                0, 10, (self.x.shape[1],)).astype(self.dtype)
        else:
            self.class_weight = None

    def check_forward(self, x_data, t_data, class_weight):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        loss = functions.softmax_cross_entropy(
            x, t, cache_score=self.cache_score, class_weight=class_weight,
            reduce='no')
        self.assertEqual(loss.shape, t_data.shape)
        self.assertEqual(loss.data.dtype, self.dtype)
        self.assertEqual(hasattr(loss.creator, 'y'), self.cache_score)
        loss_value = cuda.to_cpu(loss.data)

        x = numpy.rollaxis(self.x, 1, self.x.ndim).reshape(
            (self.t.size, self.x.shape[1]))
        t = self.t.ravel()
        l = loss_value.ravel()
        for xi, ti, li in six.moves.zip(x, t, l):
            if ti == -1:
                continue
            log_z = numpy.ufunc.reduce(numpy.logaddexp, xi)
            if class_weight is None:
                loss_expect = -(xi - log_z)[ti]
            else:
                loss_expect = -(xi - log_z)[ti] * class_weight[ti]

            testing.assert_allclose(
                loss_expect, li, **self.check_forward_options)

    @condition.retry(3)
    def test_forward_cpu(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            self.check_forward(self.x, self.t, self.class_weight)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        if not self.weight_apply:
            weight = None
        else:
            weight = cuda.to_gpu(self.class_weight)
        with chainer.using_config('use_cudnn', self.use_cudnn):
            self.check_forward(
                cuda.to_gpu(self.x), cuda.to_gpu(self.t), weight)

    def check_backward(
            self, x_data, t_data, g_data, class_weight):
        func = functions.SoftmaxCrossEntropy(
            cache_score=self.cache_score,
            class_weight=class_weight, reduce='no')
        gradient_check.check_backward(
            func, (x_data, t_data), g_data, eps=0.02,
            **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            self.check_backward(self.x, self.t, self.g, self.class_weight)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        if not self.weight_apply:
            weight = None
        else:
            weight = cuda.to_gpu(self.class_weight)
        with chainer.using_config('use_cudnn', self.use_cudnn):
            self.check_backward(
                cuda.to_gpu(self.x), cuda.to_gpu(self.t), cuda.to_gpu(self.g),
                weight)


@testing.parameterize(*testing.product({
    'use_cudnn': ['always', 'auto', 'never'],
    'normalize': [True, False],
    'cache_score': [True, False],
}))
class TestSoftmaxCrossEntropyInvalidReduce(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3)).astype('f')
        self.t = numpy.zeros((2,), 'i')

    def check_invalid_reduce(self, x, t):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            with self.assertRaises(ValueError):
                functions.softmax_cross_entropy(
                    x, t, self.normalize, self.cache_score,
                    reduce='unknown_reduce_type')

    def test_invalid_reduce_cpu(self):
        self.check_invalid_reduce(self.x, self.t)

    @attr.gpu
    def test_invalid_reduce_gpu(self):
        self.check_invalid_reduce(cuda.to_gpu(self.x), cuda.to_gpu(self.t))


@testing.parameterize(*testing.product({
    'reduce': ['mean', 'no'],
    'class_weight': [None, numpy.ones((3,), dtype=numpy.float32)]})
)
class TestNonDefaultIgnoreLabel(unittest.TestCase):

    def setUp(self):
        self.ignore_label = -2
        self.x = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)
        self.t = numpy.full((2,), self.ignore_label, dtype=numpy.int32)
        if self.reduce == 'mean':
            gy_shape = ()
        else:
            gy_shape = (2,)
        self.gy = numpy.random.uniform(-1, 1, gy_shape).astype(numpy.float32)

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
            ignore_label=self.ignore_label)
        if self.reduce == 'mean':
            expect = 0.
        else:
            expect = numpy.zeros((2,), dtype=numpy.float32)
        testing.assert_allclose(loss.data, expect)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(numpy)

    @attr.gpu
    @condition.retry(3)
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
        f = functions.SoftmaxCrossEntropy(
            reduce=self.reduce, class_weight=class_weight,
            ignore_label=self.ignore_label)
        gradient_check.check_backward(f, (x, t), gy)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(numpy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.cupy)


testing.run_module(__name__, __file__)
