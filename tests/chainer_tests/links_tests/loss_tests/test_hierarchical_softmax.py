import copy
import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import links
from chainer.links.loss import hierarchical_softmax
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


class TestTreeParser(unittest.TestCase):

    def setUp(self):
        self.parser = hierarchical_softmax.TreeParser()

    def test_parse(self):
        # Internal nodes are numbered in pre-order.
        # ((0 (1 2)) (3 4))
        # -->
        # (#0 (#1 0 (#2 1 2) (#3 3 4))
        tree = ((0, (1, 2)), (3, 4))
        self.parser.parse(tree)
        self.assertEqual(self.parser.size(), 4)
        paths = self.parser.get_paths()
        codes = self.parser.get_codes()
        numpy.testing.assert_array_equal(paths[0], [0, 1])
        numpy.testing.assert_array_equal(codes[0], [1, 1])
        numpy.testing.assert_array_equal(paths[1], [0, 1, 2])
        numpy.testing.assert_array_equal(codes[1], [1, -1, 1])
        numpy.testing.assert_array_equal(paths[2], [0, 1, 2])
        numpy.testing.assert_array_equal(codes[2], [1, -1, -1])
        numpy.testing.assert_array_equal(paths[3], [0, 3])
        numpy.testing.assert_array_equal(codes[3], [-1, 1])
        numpy.testing.assert_array_equal(paths[4], [0, 3])
        numpy.testing.assert_array_equal(codes[4], [-1, -1])

        self.assertEqual(
            self.parser.get_parent2child(),
            {0: (1, 3), 1: (-1, 2), 2: (-1, -1), 3: (-1, -1)})
        self.assertEqual(
            self.parser.get_node2word(),
            {0: (-1, -1), 1: (0, -1), 2: (1, 2), 3: (3, 4)})


class TestBinaryHierarchicalSoftmax(unittest.TestCase):

    def setUp(self):
        tree = ((0, 1), ((2, 3), 4))
        self.link = links.BinaryHierarchicalSoftmax(3, tree)
        self.link.cleargrads()
        self.x = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)
        self.t = numpy.array([0, 2]).astype(numpy.int32)
        self.gy = numpy.random.uniform(-1, 1, ()).astype(numpy.float32)

        self.W = self.link.W.data.copy()

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

    def check_sample(self, x):
        x = chainer.Variable(x)
        result = self.link.sample(x)
        self.assertIsInstance(result, self.link.xp.ndarray)
        self.assertEqual(len(result), x.shape[0])
        for word_id in result:
            self.assertEqual(word_id.dtype, self.link.xp.int32)
            # Check if the result is in the valid range
            self.assertLessEqual(0, word_id)
            self.assertLessEqual(word_id, self.max_value_tree)

    def test_sample_cpu(self):
        x = numpy.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]], numpy.float32)
        self.check_sample(x)

    @attr.gpu
    def test_sample_gpu(self):
        x = numpy.array([[1.0, 2.0, 3.0]], numpy.float32)
        self.link.to_gpu()
        self.check_sample(cuda.to_gpu(x))


testing.run_module(__name__, __file__)
