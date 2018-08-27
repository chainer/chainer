import math
import unittest

import numpy
import six

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer import utils


@testing.parameterize(*(testing.product({
    # Test dtype
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'shape': [(8, 7)],
    'normalize': [True],
    'label_dtype': [numpy.int32],
}) + [
    # Test normalize
    {'dtype': numpy.float32,
     'shape': (8, 7),
     'normalize': False,
     'label_dtype': numpy.int32,
     },
    # Test ignore_all
    {'dtype': numpy.float32,
     'shape': (8, 7),
     'normalize': True,
     'ignore_all': True,
     'label_dtype': numpy.int32,
     },
    # too large shape causes int32 -> float64 issue
    {'dtype': numpy.float32,
     'shape': (65536, 1),
     'normalize': False,
     'label_dtype': numpy.int32,
     },
] + testing.product({
    # Test label_dtype
    'dtype': [numpy.float32],
    'shape': [(8, 7)],
    'normalize': [True],
    'label_dtype': [numpy.int8, numpy.int16, numpy.int32, numpy.int64],
})))
class TestSigmoidCrossEntropy(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        if getattr(self, 'ignore_all', False):
            self.t = -numpy.ones(self.shape).astype(self.label_dtype)
        else:
            self.t = numpy.random.randint(-1, 2,
                                          self.shape).astype(self.label_dtype)
        self.gy = numpy.random.random(self.shape).astype(self.dtype)
        self.ggx = numpy.random.uniform(
            -1, 1, self.shape).astype(self.dtype)

        if self.dtype == numpy.float16:
            self.places = 2
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 5e-2, 'rtol': 5e-2}
            self.check_double_backward_options = {
                'dtype': numpy.float64, 'atol': 5e-2, 'rtol': 5e-2}
        else:
            self.places = 5
            self.check_backward_options = {'atol': 5e-3, 'rtol': 5e-3}
            self.check_double_backward_options = {'atol': 5e-3, 'rtol': 5e-3}

    def check_forward(self, x_data, t_data, use_cudnn='always'):
        x_val = chainer.Variable(x_data)
        t_val = chainer.Variable(t_data)
        with chainer.using_config('use_cudnn', use_cudnn):
            loss = functions.sigmoid_cross_entropy(x_val, t_val,
                                                   self.normalize)
        self.assertEqual(loss.data.shape, ())
        self.assertEqual(loss.data.dtype, self.dtype)
        loss_value = float(cuda.to_cpu(loss.data))

        # Compute expected value
        loss_expect = 0
        non_ignore_count = 0
        for i in six.moves.range(self.x.shape[0]):
            for j in six.moves.range(self.x.shape[1]):
                xd, td = self.x[i, j], self.t[i, j]
                if td == -1:
                    continue
                loss_expect -= xd * (td - (xd >= 0)) \
                    - math.log(1 + math.exp(-numpy.abs(xd)))
                non_ignore_count += 1
        if non_ignore_count == 0:
            loss_expect = 0
        elif self.normalize:
            loss_expect /= non_ignore_count
        else:
            loss_expect /= self.t.shape[0]
        self.assertAlmostEqual(loss_expect, loss_value, places=self.places)

    def check_forward_no_reduction(self, x_data, t_data):
        x_val = chainer.Variable(x_data)
        t_val = chainer.Variable(t_data)
        loss = functions.sigmoid_cross_entropy(
            x_val, t_val, self.normalize, reduce='no')
        self.assertEqual(loss.data.shape, self.x.shape)
        self.assertEqual(loss.data.dtype, self.dtype)
        loss_value = cuda.to_cpu(loss.data)

        # Compute expected value
        if not getattr(self, 'ignore_all', False):
            for i in six.moves.range(self.x.shape[0]):
                for j in six.moves.range(self.x.shape[1]):
                    xd, td = self.x[i, j], self.t[i, j]
                    if td == -1:
                        loss_expect = 0
                    else:
                        loss_expect = -(
                            xd * (td - (xd >= 0)) -
                            math.log(1 + math.exp(-numpy.abs(xd))))
                    self.assertAlmostEqual(
                        loss_expect, loss_value[i, j], places=self.places)

    def test_forward_cpu(self):
        with chainer.using_config('use_cudnn', 'always'):
            self.check_forward(self.x, self.t)

    def test_forward_no_reduction_cpu(self):
        with chainer.using_config('use_cudnn', 'always'):
            self.check_forward_no_reduction(self.x, self.t)

    @attr.gpu
    def test_forward_gpu(self):
        with chainer.using_config('use_cudnn', 'always'):
            self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t))

    @attr.gpu
    def test_forward_no_reduction_gpu(self):
        with chainer.using_config('use_cudnn', 'always'):
            self.check_forward_no_reduction(
                cuda.to_gpu(self.x), cuda.to_gpu(self.t))

    @attr.gpu
    def test_forward_gpu_no_cudnn(self):
        with chainer.using_config('use_cudnn', 'never'):
            self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t))

    @attr.gpu
    def test_forward_no_reduction_gpu_no_cudnn(self):
        with chainer.using_config('use_cudnn', 'never'):
            self.check_forward_no_reduction(
                cuda.to_gpu(self.x), cuda.to_gpu(self.t))

    def check_backward(self, x_data, t_data):
        # Skip too large case. That requires a long time.
        if self.shape[0] == 65536:
            return

        gradient_check.check_backward(
            functions.sigmoid_cross_entropy,
            (x_data, t_data), None, **self.check_backward_options)

    def check_backward_no_reduction(
            self, x_data, t_data, y_grad):
        # Skip too large case. That requires a long time.
        if self.shape[0] == 65536:
            return

        def f(x, t):
            return chainer.functions.sigmoid_cross_entropy(x, t, reduce='no')

        gradient_check.check_backward(
            f, (x_data, t_data), y_grad, **self.check_backward_options)

    def test_backward_cpu(self):
        with chainer.using_config('use_cudnn', 'never'):
            self.check_backward(self.x, self.t)

    def test_backward_no_reduction_cpu(self):
        with chainer.using_config('use_cudnn', 'never'):
            self.check_backward_no_reduction(self.x, self.t, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        with chainer.using_config('use_cudnn', 'always'):
            self.check_backward(
                cuda.to_gpu(self.x), cuda.to_gpu(self.t))

    @attr.gpu
    def test_backward_no_reduction_gpu(self):
        with chainer.using_config('use_cudnn', 'always'):
            self.check_backward_no_reduction(
                cuda.to_gpu(self.x), cuda.to_gpu(self.t), cuda.to_gpu(self.gy))

    @attr.gpu
    def test_backward_gpu_no_cudnn(self):
        with chainer.using_config('use_cudnn', 'never'):
            self.check_backward(
                cuda.to_gpu(self.x), cuda.to_gpu(self.t))

    @attr.gpu
    def test_backward_no_reduction_gpu_no_cudnn(self):
        with chainer.using_config('use_cudnn', 'never'):
            self.check_backward_no_reduction(
                cuda.to_gpu(self.x), cuda.to_gpu(self.t),
                cuda.to_gpu(self.gy))

    def check_double_backward(self, x_data, t_data, y_grad, gx_grad,
                              normalize=True, reduce='mean'):
        # Skip too large case. That requires a long time.
        if self.shape[0] == 65536:
            return

        if reduce == 'mean':
            y_grad = utils.force_array(y_grad.sum())

        def f(x, t):
            return chainer.functions.sigmoid_cross_entropy(
                x, t, normalize=normalize, reduce=reduce)

        gradient_check.check_double_backward(
            f, (x_data, t_data), y_grad, (gx_grad,),
            **self.check_double_backward_options)

    def test_double_backward_cpu(self):
        with chainer.using_config('use_cudnn', 'never'):
            self.check_double_backward(self.x, self.t, self.gy, self.ggx,
                                       normalize=self.normalize, reduce='mean')

    @attr.gpu
    def test_double_backward_gpu(self):
        with chainer.using_config('use_cudnn', 'always'):
            self.check_double_backward(cuda.to_gpu(self.x),
                                       cuda.to_gpu(self.t),
                                       cuda.to_gpu(self.gy),
                                       cuda.to_gpu(self.ggx),
                                       normalize=self.normalize,
                                       reduce='mean')

    def test_double_backward_no_reduction_cpu(self):
        with chainer.using_config('use_cudnn', 'never'):
            self.check_double_backward(self.x, self.t, self.gy, self.ggx,
                                       normalize=self.normalize, reduce='no')

    @attr.gpu
    def test_double_backward_no_reduction_gpu(self):
        with chainer.using_config('use_cudnn', 'always'):
            self.check_double_backward(cuda.to_gpu(self.x),
                                       cuda.to_gpu(self.t),
                                       cuda.to_gpu(self.gy),
                                       cuda.to_gpu(self.ggx),
                                       normalize=self.normalize,
                                       reduce='no')


@testing.parameterize(
    {'use_cudnn': 'always'},
    {'use_cudnn': 'auto'},
    {'use_cudnn': 'never'},
)
@attr.cudnn
class TestSigmoidCrossEntropyCudnnCall(unittest.TestCase):

    def setUp(self):
        self.x = cuda.cupy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)
        self.t = cuda.cupy.random.randint(0, 3, (4, 3)).astype(numpy.int32)
        with chainer.using_config('use_cudnn', self.use_cudnn):
            self.expect = chainer.should_use_cudnn('==always')

    def forward(self):
        x = chainer.Variable(self.x)
        t = chainer.Variable(self.t)
        return functions.sigmoid_cross_entropy(x, t)

    # Note that SigmoidCrossEntropy does not use cudnn on forward

    def test_call_cudnn_backward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            y = self.forward()
            with testing.patch('cupy.cudnn.activation_forward') as func:
                y.backward()
                self.assertEqual(func.called, self.expect)


testing.run_module(__name__, __file__)
