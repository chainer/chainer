import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
#from chainer.functions.activation import slstm
from chainer import gradient_check
from chainer import testing
from chainer.testing import backend

def _sigmoid(x):
    half = x.dtype.type(0.5)
    return numpy.tanh(x * half) * half + half

def inject_backend_tests(method_names):
    decorator = backend.inject_backend_tests(
        method_names,
        # CPU tests
        testing.product({
            'use_cuda': [False],
            'use_ideep': ['never', 'always'],
        })
        # GPU tests
        + [{'use_cuda': True}])
    return decorator

#@testing.parameterize(*(testing.product({
#    'batch': [3, 2, 0],
#    'dtype': [numpy.float32],
#}) + testing.product({
#    'batch': [3],
#    'dtype': [numpy.float16, numpy.float32, numpy.float64],
#})))
@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
#@testing.parameterize(*testing.product({
#    'dtype': [numpy.float32],
#}))
#@testing.fix_random()
@inject_backend_tests([
    'test_forward',
    'test_flat_forward',
    'test_full_backward',
    'test_flat_full_backward',
    'test_no_gc_backward',
    'test_flat_no_gc_backward',
    'test_no_gh_backward',
    'test_flat_no_gh_backward',
    'test_double_backward'])
class TestSLSTM(unittest.TestCase):

    def setUp(self):
        c_prev1 = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(self.dtype)
        c_prev2 = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(self.dtype)
        x1 = numpy.random.uniform(-1, 1, (3, 8, 4)).astype(self.dtype)
        x2 = numpy.random.uniform(-1, 1, (3, 8, 4)).astype(self.dtype)

        gc = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(self.dtype)
        gh = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(self.dtype)

        ggc_prev1 = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(self.dtype)
        ggc_prev2 = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(self.dtype)
        ggx1 = numpy.random.uniform(-1, 1, (3, 8, 4)).astype(self.dtype)
        ggx2 = numpy.random.uniform(-1, 1, (3, 8, 4)).astype(self.dtype)

        self.inputs = [c_prev1, c_prev2, x1, x2]
        self.grad_outputs = [gc, gh]
        self.grad_grad_inputs = [ggc_prev1, ggc_prev2, ggx1, ggx2]

        self.check_forward_options = {}
        self.check_backward_options = {'dtype': numpy.float64}
        self.check_double_backward_options = {'dtype': numpy.float64}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-3, 'rtol': 1e-2}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 5e-3, 'rtol': 5e-2}
            self.check_double_backward_options = {
                'dtype': numpy.float64, 'atol': 5e-3, 'rtol': 5e-2}

    def flat(self, arrays):
        return [None if a is None else a[:, :, 0] for a in arrays]
#        self.c_prev1 = self.c_prev1[:, :, 0].copy()
#        self.c_prev2 = self.c_prev2[:, :, 0].copy()
#        self.x1 = self.x1[:, :, 0].copy()
#        self.x2 = self.x2[:, :, 0].copy()
#        self.gc = self.gc[:, :, 0].copy()
#        self.gh = self.gh[:, :, 0].copy()

    def forward_cpu(self, inputs):
        c_prev1, c_prev2, x1 ,x2 = inputs
        batch = x1.shape[0]
        a1_in = x1[:, [0, 4]]
        i1_in = x1[:, [1, 5]]
        f1_in = x1[:, [2, 6]]
        o1_in = x1[:, [3, 7]]
        a2_in = x2[:, [0, 4]]
        i2_in = x2[:, [1, 5]]
        f2_in = x2[:, [2, 6]]
        o2_in = x2[:, [3, 7]]
        c_expect = _sigmoid(i1_in) * numpy.tanh(a1_in) + \
            _sigmoid(i2_in) * numpy.tanh(a2_in) + \
            _sigmoid(f1_in) * c_prev1 + \
            _sigmoid(f2_in) * c_prev2
        h_expect = _sigmoid(o1_in + o2_in) * numpy.tanh(c_expect)
        return c_expect, h_expect
    
    def check_forward(self, inputs, backend_config):
        c_prev1, c_prev2, x1 ,x2 = inputs
        batch = x1.shape[0]
        c_expect_2 = c_prev1[batch:]
        c_expect_1, h_expect = self.forward_cpu(inputs)

        if backend_config.use_cuda:
            inputs = cuda.to_gpu(inputs)
        inputs = [chainer.Variable(xx) for xx in inputs]

        with backend_config:
            c, h = functions.slstm(*inputs)
            assert c.data.dtype == self.dtype
            assert h.data.dtype == self.dtype

        testing.assert_allclose(
            c_expect_1, c.data[:batch], **self.check_forward_options)
        testing.assert_allclose(
            c_expect_2, c.data[batch:], **self.check_forward_options)
        testing.assert_allclose(
            h_expect, h.data, **self.check_forward_options)

#    def test_forward_cpu(self):
#        self.check_forward(self.c_prev1, self.c_prev2, self.x1, self.x2)

#    def test_flat_forward_cpu(self):
#        self.flat()
#        self.test_forward_cpu()

    def test_forward(self, backend_config):
        self.check_forward(self.inputs, backend_config)

    def test_flat_forward(self, backend_config):
        self.check_forward(self.flat(self.inputs), backend_config)

    def check_backward(self, inputs, grad_outputs, backend_config):
        if backend_config.use_cuda:
            inputs = cuda.to_gpu(inputs)
            grad_outputs = cuda.to_gpu(grad_outputs)

        with backend_config:
            gradient_check.check_backward(
                functions.slstm, inputs, grad_outputs,
                **self.check_backward_options)

    def test_full_backward(self, backend_config):
        self.check_backward(self.inputs, self.grad_outputs, backend_config)

    def test_flat_full_backward(self, backend_config):
        self.check_backward(
            self.flat(self.inputs), self.flat(self.grad_outputs),
            backend_config)

    def test_no_gc_backward(self, backend_config):
        grad_outputs = [None, self.grad_outputs[1]]
        self.check_backward(self.inputs, grad_outputs, backend_config)

    def test_flat_no_gc_backward(self, backend_config):
        grad_outputs = [None, self.grad_outputs[1]]
        self.check_backward(
            self.flat(self.inputs), self.flat(grad_outputs), backend_config)

    def test_no_gh_backward(self, backend_config):
        grad_outputs = [self.grad_outputs[0], None]
        self.check_backward(self.inputs, grad_outputs, backend_config)

    def test_flat_no_gh_backward(self, backend_config):
        grad_outputs = [self.grad_outputs[0], None]
        self.check_backward(
            self.flat(self.inputs), self.flat(grad_outputs), backend_config)

    def check_double_backward(
            self, inputs, grad_outputs, grad_grad_inputs, backend_config):
        if backend_config.use_cuda:
            inputs = cuda.to_gpu(inputs)
            grad_outputs = cuda.to_gpu(grad_outputs)
            grad_grad_inputs = cuda.to_gpu(grad_grad_inputs)

        with backend_config:
            gradient_check.check_double_backward(
                chainer.functions.slstm, inputs, grad_outputs, grad_grad_inputs,
                **self.check_double_backward_options)

    def test_double_backward(self,  backend_config):
        self.check_double_backward(
            self.inputs, self.grad_outputs, self.grad_grad_inputs,
            backend_config)

#    def test_full_backward_cpu(self):
#        self.check_backward(self.c_prev1, self.c_prev2, self.x1, self.x2,
#                            self.gc, self.gh)

#    def test_full_backward_cpu(self, backend_config):
#        self.check_backward(self.inputs, self.grad_outputs, backend_config)

#    def test_flat_full_backward_cpu(self):
#        self.flat()
#        self.test_full_backward_cpu()

#    def test_no_gc_backward_cpu(self):
#        self.check_backward(self.c_prev1, self.c_prev2, self.x1, self.x2,
#                            None, self.gh)

#    def test_flat_no_gc_backward_cpu(self):
#        self.flat()
#        self.test_no_gc_backward_cpu()

#    def test_no_gh_backward_cpu(self):
#        self.check_backward(self.c_prev1, self.c_prev2, self.x1, self.x2,
#                            self.gc, None)

#    def test_flat_no_gh_backward_cpu(self):
#        self.flat()
#        self.test_no_gh_backward_cpu()

#    @attr.gpu
#    def test_full_backward_gpu(self):
#        self.check_backward(
#            cuda.to_gpu(self.c_prev1),
#            cuda.to_gpu(self.c_prev2),
#            cuda.to_gpu(self.x1),
#            cuda.to_gpu(self.x2),
#            cuda.to_gpu(self.gc),
#            cuda.to_gpu(self.gh))

#    @attr.gpu
#    def test_flat_full_backward_gpu(self):
#        self.flat()
#        self.test_full_backward_gpu()

#    @attr.gpu
#    def test_no_gc_backward_gpu(self):
#        self.check_backward(
#            cuda.to_gpu(self.c_prev1),
#            cuda.to_gpu(self.c_prev2),
#            cuda.to_gpu(self.x1),
#            cuda.to_gpu(self.x2),
#            None,
#            cuda.to_gpu(self.gh))

#    @attr.gpu
#    def test_flat_no_gc_backward_gpu(self):
#        self.flat()
#        self.test_no_gc_backward_gpu()

#    @attr.gpu
#    def test_no_gh_backward_gpu(self):
#        self.check_backward(
#            cuda.to_gpu(self.c_prev1),
#            cuda.to_gpu(self.c_prev2),
#            cuda.to_gpu(self.x1),
#            cuda.to_gpu(self.x2),
#            cuda.to_gpu(self.gc),
#            None)

#    @attr.gpu
#    def test_flat_no_gh_backward_gpu(self):
#        self.flat()
#        self.test_no_gh_backward_gpu()


testing.run_module(__name__, __file__)
