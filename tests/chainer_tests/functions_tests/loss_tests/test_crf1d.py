import itertools
import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product_dict(
    [
        {'lengths': [3, 3], 'batches': [2, 2, 2]},
        {'lengths': [3, 2, 1], 'batches': [3, 2, 1]},
        {'lengths': [3, 1, 1], 'batches': [3, 1, 1]},
        {'lengths': [1, 1], 'batches': [2]},
    ],
    [
        {'reduce': 'mean'},
        {'reduce': 'no'},
    ]
))
class TestCRF1d(unittest.TestCase):
    n_label = 3

    def setUp(self):
        self.cost = numpy.random.uniform(
            -1, 1, (self.n_label, self.n_label)).astype(numpy.float32)
        self.xs = [numpy.random.uniform(
            -1, 1, (b, 3)).astype(numpy.float32) for b in self.batches]
        self.ys = [
            numpy.random.randint(
                0, self.n_label, (b,)).astype(numpy.int32)
            for b in self.batches]
        self.g = numpy.random.uniform(
            -1, 1, (len(self.lengths))).astype(numpy.float32)

    def _calc_score(self, batch, ys):
        return sum(x[batch, y] for x, y in zip(self.xs, ys)) + \
            sum(self.cost[y1, y2] for y1, y2 in zip(ys[:-1], ys[1:]))

    def check_forward(self, cost_data, xs_data, ys_data):
        cost = chainer.Variable(cost_data)
        xs = [chainer.Variable(x) for x in xs_data]
        ys = [chainer.Variable(y) for y in ys_data]
        actual = functions.crf1d(cost, xs, ys, reduce=self.reduce)

        z = numpy.zeros((self.batches[0],), numpy.float32)
        for b, length in enumerate(self.lengths):
            for ys in itertools.product(range(self.n_label), repeat=length):
                z[b] += numpy.exp(self._calc_score(b, ys))

        score = numpy.zeros((self.batches[0],), numpy.float32)
        for b, length in enumerate(self.lengths):
            ys = [self.ys[i][b] for i in range(length)]
            score[b] = self._calc_score(b, ys)

        loss = -(score - numpy.log(z))
        if self.reduce == 'mean':
            expect = numpy.sum(loss) / self.batches[0]
        elif self.reduce == 'no':
            expect = loss

        testing.assert_allclose(actual.data, expect)

    def test_forward_cpu(self):
        self.check_forward(self.cost, self.xs, self.ys)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.cost),
                           [cuda.to_gpu(x) for x in self.xs],
                           [cuda.to_gpu(y) for y in self.ys])

    def check_backward(self, cost_data, xs_data, ys_data, g_data):
        def f(cost, *args):
            xs = args[:len(args) // 2]
            ys = args[len(args) // 2:]
            return functions.crf1d(cost, xs, ys, reduce=self.reduce)

        args = [cost_data] + xs_data + ys_data
        if self.reduce == 'mean':
            grad = None
        elif self.reduce == 'no':
            grad = g_data
        gradient_check.check_backward(
            f, args, grad, rtol=1e-3, atol=1e-3)

    def test_backward_cpu(self):
        self.check_backward(self.cost, self.xs, self.ys, self.g)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.cost),
                            [cuda.to_gpu(x) for x in self.xs],
                            [cuda.to_gpu(y) for y in self.ys],
                            cuda.to_gpu(self.g))

    def check_argmax(self, cost_data, xs_data):
        cost = chainer.Variable(cost_data)
        xs = [chainer.Variable(x) for x in xs_data]
        s, path = functions.loss.crf1d.argmax_crf1d(cost, xs)

        self.assertIsInstance(s, chainer.Variable)
        self.assertIsInstance(path, list)
        self.assertEqual(s.shape, (self.batches[0],))
        self.assertEqual(len(path), len(self.batches))
        for b, p in zip(self.batches, path):
            self.assertEqual(p.shape, (b,))

        best_paths = [numpy.empty((length,), numpy.int32)
                      for length in self.batches]
        for b, length in enumerate(self.lengths):
            best_path = None
            best_score = 0
            for ys in itertools.product(range(self.n_label), repeat=length):
                score = self._calc_score(b, ys)
                if best_path is None or best_score < score:
                    best_path = ys
                    best_score = score

            for i, p in enumerate(best_path):
                best_paths[i][b] = p

            testing.assert_allclose(s.data[b], best_score)

        for t in range(len(self.batches)):
            numpy.testing.assert_array_equal(
                cuda.to_cpu(path[t]), best_paths[t])

    def test_argmax_cpu(self):
        self.check_argmax(self.cost, self.xs)

    @attr.gpu
    def test_argmax_gpu(self):
        self.check_argmax(cuda.to_gpu(self.cost),
                          [cuda.to_gpu(x) for x in self.xs])

    def check_invalid_option(self, cost_data, xs_data, ys_data):
        with self.assertRaises(ValueError):
            functions.crf1d(cost_data, xs_data, ys_data, 'invalid_option')

    def test_invalid_option_cpu(self):
        self.check_invalid_option(self.cost, self.xs, self.ys)

    @attr.gpu
    def test_invalid_option_gpu(self):
        self.check_invalid_option(
            cuda.to_gpu(self.cost),
            [cuda.to_gpu(x) for x in self.xs],
            [cuda.to_gpu(y) for y in self.ys])


testing.run_module(__name__, __file__)
