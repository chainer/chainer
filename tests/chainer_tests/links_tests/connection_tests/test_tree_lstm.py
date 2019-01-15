import unittest

import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


def _sigmoid(x):
    half = x.dtype.type(0.5)
    xp = backend.get_array_module(x)
    return xp.tanh(x * half) * half + half


def _child_sum_tree_lstm(func, *inputs):
    cs = inputs[:len(inputs) // 2]
    hs = inputs[len(inputs) // 2:-1]
    x = inputs[-1]
    xp = backend.get_array_module(x)
    with cuda.get_device_from_array(x):
        W_x = func.W_x.W.data.T
        b_x = func.W_x.b.data
        W_h_aio = func.W_h_aio.W.data.T
        W_h_f = func.W_h_f.W.data.T

        W_xa, W_xi, W_xo, W_xf = xp.split(W_x, 4, 1)
        b_a, b_i, b_o, b_f = xp.split(b_x[None, ], 4, 1)
        W_ha, W_hi, W_ho = xp.split(W_h_aio, 3, 1)
        W_hf = W_h_f

        if len(hs) >= 1:
            sum_h = sum(hs)
            a = x.dot(W_xa) + sum_h.dot(W_ha) + b_a
            i = x.dot(W_xi) + sum_h.dot(W_hi) + b_i
            o = x.dot(W_xo) + sum_h.dot(W_ho) + b_o
            f_list = [x.dot(W_xf) + h.dot(W_hf) + b_f for h in hs]
        else:
            a = x.dot(W_xa) + b_a
            i = x.dot(W_xi) + b_i
            o = x.dot(W_xo) + b_o
        a = xp.tanh(a)
        i = _sigmoid(i)
        o = _sigmoid(o)

        if len(hs) >= 1:
            f_list = [_sigmoid(f) for f in f_list]
            c_next = sum([f * c for f, c in zip(f_list, cs)], a * i)
            y = o * xp.tanh(c_next)
        else:
            c_next = a * i
            y = o * xp.tanh(c_next)
    return c_next, y


def _nary_tree_lstm(func, *inputs):
    cs = inputs[:len(inputs) // 2]
    hs = inputs[len(inputs) // 2:-1]
    x = inputs[-1]
    xp = backend.get_array_module(x)
    with cuda.get_device_from_array(x):
        W_x = func.W_x.W.data.T
        b_x = func.W_x.b.data
        W_h_list = [getattr(func, 'W_h{}'.format(i)).W.data.T
                    for i in range(1, func.n_ary + 1)]

        W_xs = xp.split(W_x, 3 + func.n_ary, 1)
        W_xa, W_xi, W_xo, W_xfs = W_xs[0], W_xs[1], W_xs[2], W_xs[3:]
        b_xs = xp.split(b_x[None, ], 3 + func.n_ary, 1)
        b_a, b_i, b_o, b_fs = b_xs[0], b_xs[1], b_xs[2], b_xs[3:]
        W_ha_list = [xp.split(W_h, 3 + func.n_ary, 1)[0]
                     for W_h in W_h_list]
        W_hi_list = [xp.split(W_h, 3 + func.n_ary, 1)[1]
                     for W_h in W_h_list]
        W_ho_list = [xp.split(W_h, 3 + func.n_ary, 1)[2]
                     for W_h in W_h_list]
        W_hfs_list = [xp.split(W_h, 3 + func.n_ary, 1)[3:]
                      for W_h in W_h_list]
        assert(all(len(W_hfs_list) == len(W_hfs) for W_hfs in W_hfs_list))

        a = x.dot(W_xa) + b_a + \
            sum(h.dot(W_ha) for h, W_ha in zip(hs, W_ha_list))
        i = x.dot(W_xi) + b_i + \
            sum(h.dot(W_hi) for h, W_hi in zip(hs, W_hi_list))
        o = x.dot(W_xo) + b_o + \
            sum(h.dot(W_ho) for h, W_ho in zip(hs, W_ho_list))
        f_list = [x.dot(W_xf) + b_f +
                  sum(h.dot(W_hf) for h, W_hf in zip(hs, W_hf_list))
                  for W_xf, b_f, W_hf_list
                  in zip(W_xfs, b_fs, zip(*W_hfs_list))]

        a = xp.tanh(a)
        i = _sigmoid(i)
        o = _sigmoid(o)
        f_list = [_sigmoid(f) for f in f_list]

        c_next = a * i + sum(f * c for f, c in zip(f_list, cs))
        y = o * xp.tanh(c_next)
    return c_next, y


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32],
    'n_ary': [0, 1, 2, 3],
    'in_size': [6, 9],
    'out_size': [9],
    'model_type': ['ChildSumTreeLSTM', 'NaryTreeLSTM'],
}))
class TestTreeLSTM(unittest.TestCase):

    def setUp(self):
        if self.model_type == 'ChildSumTreeLSTM':
            self.link = links.ChildSumTreeLSTM(
                self.in_size, self.out_size)
        elif self.model_type == 'NaryTreeLSTM':
            if self.n_ary == 0:
                # n_ary=0 test should be skipped for NaryTreeLSTM
                self.n_ary = 1
            self.link = links.NaryTreeLSTM(
                self.in_size, self.out_size, n_ary=self.n_ary)
        else:
            NotImplementedError()

        for p in self.link.params():
            p.data[:] = numpy.random.uniform(-1, 1, p.shape).astype(self.dtype)

        self.c_prevs = [
            numpy.random.uniform(-1, 1, (5, self.out_size)).astype(self.dtype)
            for _ in range(self.n_ary)]
        self.h_prevs = [
            numpy.random.uniform(-1, 1, (5, self.out_size)).astype(self.dtype)
            for _ in range(self.n_ary)]
        self.x = numpy.random.uniform(
            -1, 1, (5, self.in_size)).astype(self.dtype)
        self.inputs = self.c_prevs + self.h_prevs + [self.x]

        self.gc = numpy.random.uniform(
            -1, 1, (5, self.out_size)).astype(self.dtype)
        self.gh = numpy.random.uniform(
            -1, 1, (5, self.out_size)).astype(self.dtype)

        self.check_forward_options = {}
        self.check_backward_options = {'dtype': numpy.float64}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 5e-4, 'rtol': 5e-3}

    def check_forward(self, *inputs_data):
        inputs_variable = [chainer.Variable(v) for v in inputs_data]

        c, h = self.link(*inputs_variable)
        self.assertEqual(c.data.dtype, self.dtype)
        self.assertEqual(h.data.dtype, self.dtype)

        # Compute expected out
        if self.model_type == 'ChildSumTreeLSTM':
            c_expect, h_expect = _child_sum_tree_lstm(self.link, *inputs_data)
        elif self.model_type == 'NaryTreeLSTM':
            c_expect, h_expect = _nary_tree_lstm(self.link, *inputs_data)
        else:
            NotImplementedError()

        testing.assert_allclose(
            c_expect, c.data, **self.check_forward_options)
        testing.assert_allclose(
            h_expect, h.data, **self.check_forward_options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(*self.inputs)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.link.to_gpu()
        self.check_forward(*[cuda.to_gpu(v) for v in self.inputs])

    @attr.multi_gpu(2)
    def test_forward_gpu_multi(self):
        with cuda.get_device_from_id(0):
            self.link.to_gpu()
            inputs = [cuda.to_gpu(v) for v in self.inputs]
        with cuda.get_device_from_id(1):
            self.check_forward(*inputs)

    def check_forward_valid_none(self, *inputs_data):
        inputs_variable = [chainer.Variable(v)
                           if v is not None else v for v in inputs_data]
        xp = self.link.xp
        inputs_data = [xp.zeros(self.h_prevs[0].shape, dtype=self.dtype)
                       if v is None else v for v in inputs_data[:-1]] + \
            [xp.zeros(self.x.shape, dtype=self.dtype)
             if inputs_data[-1] is None else inputs_data[-1]]

        if self.n_ary == 0:
            # in this case for link(x) without cs and hs,
            # it does not include any None.
            pass
        else:
            c, h = self.link(*inputs_variable)
            self.assertEqual(c.data.dtype, self.dtype)
            self.assertEqual(h.data.dtype, self.dtype)

            # Compute expected out
            if self.model_type == 'ChildSumTreeLSTM':
                c_expect, h_expect = _child_sum_tree_lstm(
                    self.link, *inputs_data)
            elif self.model_type == 'NaryTreeLSTM':
                c_expect, h_expect = _nary_tree_lstm(self.link, *inputs_data)
            else:
                NotImplementedError()

            testing.assert_allclose(
                c_expect, c.data, **self.check_forward_options)
            testing.assert_allclose(
                h_expect, h.data, **self.check_forward_options)

    def test_forward_none_ch_cpu(self):
        inputs = [None] * len(self.c_prevs) + \
                 [None] * len(self.h_prevs) + [self.x]
        self.check_forward_valid_none(*inputs)

    @attr.gpu
    def test_forward_none_ch_gpu(self):
        self.link.to_gpu()
        inputs = [None] * len(self.c_prevs) + \
                 [None] * len(self.h_prevs) + \
                 [cuda.to_gpu(self.x)]
        self.check_forward_valid_none(*inputs)

    def test_forward_none_x_cpu(self):
        inputs = self.c_prevs + self.h_prevs + [None]
        self.check_forward_valid_none(*inputs)

    @attr.gpu
    def test_forward_none_x_gpu(self):
        self.link.to_gpu()
        inputs = [cuda.to_gpu(v) for v in self.c_prevs] + \
                 [cuda.to_gpu(v) for v in self.h_prevs] + [None]
        self.check_forward_valid_none(*inputs)

    def check_forward_invalid_none(self, *inputs_data):
        inputs_variable = [chainer.Variable(v)
                           if v is not None else v for v in inputs_data]
        self.assertRaises(ValueError, self.link, *inputs_variable)

    def test_forward_none_chx_cpu(self):
        inputs = [None] * len(self.inputs)
        self.check_forward_invalid_none(*inputs)

    @attr.gpu
    def test_forward_none_chx_gpu(self):
        self.link.to_gpu()
        inputs = [None] * len(self.inputs)
        self.check_forward_invalid_none(*inputs)

    def check_backward(self, c_grad, h_grad, *inputs):
        gradient_check.check_backward(
            self.link,
            inputs,
            (c_grad, h_grad),
            **self.check_backward_options)

    @condition.retry(3)
    def test_full_backward_cpu(self):
        self.check_backward(self.gc, self.gh, *self.inputs)

    @condition.retry(3)
    def test_no_gc_backward_cpu(self):
        self.check_backward(None, self.gh, *self.inputs)

    @condition.retry(3)
    def test_no_gh_backward_cpu(self):
        self.check_backward(self.gc, None, *self.inputs)

    @attr.gpu
    @condition.retry(3)
    def test_full_backward_gpu(self):
        self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.gc), cuda.to_gpu(self.gh),
                            *[cuda.to_gpu(v) for v in self.inputs])

    @attr.gpu
    @condition.retry(3)
    def test_no_gc_backward_gpu(self):
        self.link.to_gpu()
        self.check_backward(None, cuda.to_gpu(self.gh),
                            *[cuda.to_gpu(v) for v in self.inputs])

    @attr.gpu
    @condition.retry(3)
    def test_no_gh_backward_gpu(self):
        self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.gc), None,
                            *[cuda.to_gpu(v) for v in self.inputs])


testing.run_module(__name__, __file__)
