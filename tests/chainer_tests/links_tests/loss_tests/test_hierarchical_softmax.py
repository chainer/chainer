import copy
import unittest

import numpy
import six

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
        self.link = links.BinaryHierarchicalSoftmax(3, tree)
        self.link.cleargrads()
        self.x = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)
        self.t = numpy.array([0, 2]).astype(numpy.int32)
        self.gy = numpy.random.uniform(-1, 1, ()).astype(numpy.float32)

        self.W = self.link.W.data.copy()
        _func = copy.deepcopy(self.link)._func
        self.paths = _func.paths
        self.codes = _func.codes
        self.begins = _func.begins

        # Compute max value in `tree`
        self.max_value_tree = 0
        self.compute_max_value_in_tree(tree)

    def compute_max_value_in_tree(self, node):
        if isinstance(node, tuple):
            left, right = node
            self.compute_max_value_in_tree(left)
            self.compute_max_value_in_tree(right)
        else:
            if self.max_value_tree < node:
                self.max_value_tree = node

    def check_sum(self, x, gpu=False):
        total = 0
        for i in range(5):
            t = numpy.array([i], dtype=numpy.int32)
            if gpu:
                t = cuda.to_gpu(t)
            loss = self.link(chainer.Variable(x), chainer.Variable(t)).data
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
            cpu_loss, cuda.to_cpu(gpu_loss))

    def check_backward(self, x_data, t_data, y_grad):
        gradient_check.check_backward(
            self.link, (x_data, t_data), y_grad, self.link.W,
            atol=1e-4, rtol=1e-3)

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

    def compute_argmax(self, x, gpu):
        if gpu:
            x = cuda.to_cpu(x)
        batchsize = x.shape[0]
        expercted_result = []
        for i in six.moves.range(batchsize):
            scores = self.compute_score_manually(x[i], gpu)
            expect = numpy.argmax(scores).astype('i')
            expect = numpy.array([expect], dtype='i')
            if gpu:
                expect = cuda.to_gpu(expect)
            expercted_result.append(expect)
        return expercted_result

    def node_score(self, node_index, x, code):
        return 1 / (numpy.exp(- self.W[node_index].dot(x) * code) + 1)

    def compute_score_manually(self, x, gpu):
        # tree = ((0, 1), ((2, 3), 4))
        scores = numpy.zeros((5, ), dtype='f')
        scores[0] = self.node_score(0, x, 1) * self.node_score(1, x, 1)
        scores[1] = self.node_score(0, x, 1) * self.node_score(1, x, -1)
        scores[2] = self.node_score(0, x, -1) * self.node_score(2, x, 1) * \
            self.node_score(3, x, 1)
        scores[3] = self.node_score(0, x, -1) * self.node_score(2, x, 1) * \
            self.node_score(3, x, -1)
        scores[4] = self.node_score(0, x, -1) * self.node_score(2, x, -1)
        return scores

    def check_argmax(self, x, gpu=False):
        # Compute correct argmax word index.

        expercted_result = self.compute_argmax(x, gpu)

        x = chainer.Variable(x)
        result = self.link.argmax(x)
        self.assertIsInstance(result, self.link.xp.ndarray)
        self.assertEqual(len(result), x.shape[0])
        for word_id, expect in six.moves.zip(result, expercted_result):
            self.assertEqual(word_id.dtype, self.link.xp.int32)
            # Check if the result is in the valid range
            self.assertLessEqual(0, word_id)
            self.assertLessEqual(word_id, self.max_value_tree)
            # Check if result is equal to expect result.
            self.assertEqual(expect, word_id)

    def test_argmax_cpu(self):
        x = numpy.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]], numpy.float32)
        self.check_argmax(x, gpu=False)

    @attr.gpu
    def test_argmax_gpu(self):
        x = numpy.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]], numpy.float32)
        self.link.to_gpu()
        self.check_argmax(cuda.to_gpu(x), gpu=True)


testing.run_module(__name__, __file__)
