import copy
import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


class TestHuffmanTree(unittest.TestCase):

    def test_empty(self):
        with self.assertRaises(ValueError):
            links.BinaryHierarchicalSoftmax.create_huffman_tree({})

    def test_simple(self):
        tree = links.BinaryHierarchicalSoftmax.create_huffman_tree(
            {'x': 8, 'y': 6, 'z': 5, 'w': 4, 'v': 3})
        expect = (('z', 'y'), (('v', 'w'), 'x'))
        self.assertEqual(expect, tree)

    def test_same_count(self):
        tree = links.BinaryHierarchicalSoftmax.create_huffman_tree(
            {'x': 1, 'y': 2, 'z': 3})
        # Order of the same items are not defined.
        self.assertTrue((('x', 'y'), 'z') == tree or
                        ('z', ('x', 'y')) == tree)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestBinaryHierarchicalSoftmax(unittest.TestCase):

    def setUp(self):
        self._config_user = chainer.using_config('dtype', self.dtype)
        self._config_user.__enter__()

        tree = ((0, 1), ((2, 3), 4))
        self.link = links.BinaryHierarchicalSoftmax(3, tree)
        self.link.cleargrads()
        self.x = numpy.random.uniform(-1, 1, (2, 3)).astype(self.dtype)
        self.t = numpy.array([0, 2]).astype(numpy.int32)
        self.gy = numpy.random.uniform(-1, 1, ()).astype(self.dtype)

        self.W = self.link.W.data.copy()

        if self.dtype == numpy.float16:
            self.check_sum_options = {'delta': 1e-3}
            self.test_forward_options = {'atol': 0.005}
            self.check_backward_options = {'dtype': numpy.float64}
        else:
            self.check_sum_options = {'delta': 1e-5}
            self.test_forward_options = {}
            self.check_backward_options = {}

    def tearDown(self):
        self._config_user.__exit__(None, None, None)

    def check_sum(self, x, gpu=False):
        total = 0
        for i in range(5):
            t = numpy.array([i], dtype=numpy.int32)
            if gpu:
                t = cuda.to_gpu(t)
            loss = self.link(chainer.Variable(x), chainer.Variable(t)).data
            self.assertEqual(loss.dtype, self.dtype)
            self.assertEqual(loss.shape, ())
            total += numpy.exp(-cuda.to_cpu(loss))
        self.assertAlmostEqual(1.0, float(total), **self.check_sum_options)

    @condition.retry(3)
    def test_sum_cpu(self):
        x = numpy.array([[1.0, 2.0, 3.0]], self.dtype)
        self.check_sum(x)

    @attr.gpu
    @condition.retry(3)
    def test_sum_gpu(self):
        x = numpy.array([[1.0, 2.0, 3.0]], self.dtype)
        self.link.to_gpu()
        self.check_sum(cuda.to_gpu(x), gpu=True)

    @attr.gpu
    def test_forward(self):
        # TODO(unno): We need to test return values of forward function.
        cpu_loss = self.link(chainer.Variable(self.x),
                             chainer.Variable(self.t)).data
        self.link.to_gpu()
        gpu_loss = self.link(chainer.Variable(cuda.to_gpu(self.x)),
                             chainer.Variable(cuda.to_gpu(self.t))).data
        testing.assert_allclose(
            cpu_loss, cuda.to_cpu(gpu_loss), **self.test_forward_options)

    def check_backward(self, x_data, t_data, y_grad):

        def f(x, t):
            # Force to cast the dtype of an internal state so that we can
            # perform numerical gradient in higher precision.
            if self.dtype == numpy.float16 and x.dtype == numpy.float64:
                self.link._func.codes = self.link._func.codes.astype(x.dtype)
            return self.link(x, t)

        gradient_check.check_backward(
            f, (x_data, t_data), y_grad, self.link.W,
            atol=1e-4, rtol=1e-3, **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.t, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.x),
                            cuda.to_gpu(self.t),
                            cuda.to_gpu(self.gy))

    @attr.gpu
    def test_to_cpu(self):
        f = copy.deepcopy(self.link)._func
        self.link.to_gpu()
        self.link.to_cpu()
        g = self.link._func

        self.assertTrue((f.begins == g.begins).all())
        self.assertTrue((f.paths == g.paths).all())
        self.assertTrue((f.codes == g.codes).all())


testing.run_module(__name__, __file__)
