import copy
import unittest

import numpy

import chainer
from chainer import cuda
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


class TestBinaryHierarchicalSoftmax(unittest.TestCase):

    def setUp(self):
        tree = ((0, 1), ((2, 3), 4))
        self.func = links.BinaryHierarchicalSoftmax(3, tree)
        self.func.zerograds()
        self.x = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)
        self.t = numpy.array([0, 2]).astype(numpy.int32)
        self.gy = numpy.random.uniform(-1, 1, ()).astype(numpy.float32)

        self.W = self.func.params['W'].data.copy()

    def check_sum(self, x, gpu=False):
        xp = cuda.cupy if gpu else numpy
        total = 0
        if gpu:
            self.func.to_gpu()
        for i in range(5):
            t = xp.array([i], dtype=numpy.int32)
            loss = self.func(chainer.Variable(x), chainer.Variable(t)).data
            self.assertEqual(loss.dtype, numpy.float32)
            self.assertEqual(loss.shape, ())
            total += numpy.exp(-cuda.to_cpu(loss))
        self.assertAlmostEqual(1.0, float(total), delta=1.0e-5)

    @condition.retry(3)
    def test_sum_cpu(self):
        x = numpy.array([[1.0, 2.0, 3.0]], numpy.float32)
        self.check_sum(x)

    @attr.gpu
    @condition.retry(3)
    def test_sum_gpu(self):
        x = numpy.array([[1.0, 2.0, 3.0]], numpy.float32)
        self.func.to_gpu()
        self.check_sum(cuda.to_gpu(x), gpu=True)

    @attr.gpu
    def test_forward(self):
        # TODO(unno): We need to test return values of forward function.
        cpu_loss = self.func(chainer.Variable(self.x),
                             chainer.Variable(self.t)).data
        self.func.to_gpu()
        gpu_loss = self.func(
            chainer.Variable(cuda.to_gpu(self.x)),
            chainer.Variable(cuda.to_gpu(self.t))).data
        gradient_check.assert_allclose(
            cpu_loss, cuda.to_cpu(gpu_loss))

    def check_backward(self, x_data, t_data, y_grad):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        W = self.func.params['W']

        y = self.func(x, t)
        y.grad = y_grad
        y.backward()

        f = lambda: self.func(x, t)
        gx, gW = gradient_check.numerical_grad(
            f, (x.data, W.data), (y.grad,), eps=1e-2)

        gradient_check.assert_allclose(cuda.to_cpu(gx), cuda.to_cpu(x.grad))
        gradient_check.assert_allclose(
            cuda.to_cpu(gW), cuda.to_cpu(W.grad))

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.t, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.func.to_gpu()
        self.check_backward(cuda.to_gpu(self.x),
                            cuda.to_gpu(self.t),
                            cuda.to_gpu(self.gy))

    @attr.gpu
    def test_to_cpu(self):
        f = copy.deepcopy(self.func)
        self.func.to_gpu()
        self.func.to_cpu()

        f1 = f._func
        f2 = self.func._func
        self.assertTrue((f1.begins == f2.begins).all())
        self.assertTrue((f1.paths == f2.paths).all())
        self.assertTrue((f1.codes == f2.codes).all())


testing.run_module(__name__, __file__)
