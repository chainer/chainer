import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions as F
from chainer import testing
from chainer.testing import attr


def recall(preds, ts, dtype, label_num, ignore_label):
    tp = numpy.zeros((label_num,), dtype=numpy.int32)
    support = numpy.zeros((label_num,), dtype=numpy.int32)
    for p, t in zip(preds.ravel(), ts.ravel()):
        if t == ignore_label:
            continue
        support[t] += 1
        if p == t:
            tp[t] += 1
    return dtype(tp) / support


def precision(preds, ts, dtype, label_num, ignore_label):
    tp = numpy.zeros((label_num,), dtype=numpy.int32)
    relevant = numpy.zeros((label_num,), dtype=numpy.int32)
    for p, t in zip(preds.ravel(), ts.ravel()):
        if t == ignore_label:
            continue
        relevant[p] += 1
        if p == t:
            tp[p] += 1
    return dtype(tp) / relevant


def fbeta_score(precision, recall, beta=1.0):
    beta_square = beta * beta
    return ((1 + beta_square) * precision * recall /
            (beta_square * precision + recall))


def support(ts, dtype, label_num, ignore_label):
    ret = numpy.zeros((label_num,), dtype=numpy.int32)
    for t in ts.ravel():
        if t == ignore_label:
            continue
        ret[t] += 1
    return ret


# Suppose label_num is 3 so that all valid label should be in [0, 3).\,
# then, the typical output of this function is as follows:
# has_ignore_label \ ignore_label | -1             | 0              |
#       yes                       | 0,1,-1,2,-1... | 0,1,2,2,0,1... |
#       no                        | 0,1,2,2,0,1... | 1,2,1,1,2,1... |
def make_ground_truth(label_num, shape, ignore_label, has_ignore_label, dtype):
    if (ignore_label == -1) != (has_ignore_label):  # xor
        lower = 0
    elif ignore_label == 0 and not has_ignore_label:
        lower = 1
    else:
        lower = -1

    t = numpy.random.randint(lower, label_num, shape)
    return t.astype(dtype)


@testing.parameterize(*(
    testing.product_dict(
        [{'y_shape': (100, 3), 't_shape': (100,)},
         {'y_shape': (100, 3, 5), 't_shape': (100, 5)}],
        [{'dtype': numpy.float32}],
        [{'beta': 1.0},
         {'beta': 2.0}],
        [{'label_num': 3},
         {'label_num': None}],
        [{'ignore_label': -1},
         {'ignore_label': 0}],
        [{'has_ignore_label': True},
         {'has_ignore_label': False}],
        [{'label_dtype': numpy.int32}]
    ) + testing.product_dict(
        [{'y_shape': (100, 3), 't_shape': (100,)}],
        [{'dtype': numpy.float16},
         {'dtype': numpy.float32},
         {'dtype': numpy.float64}],
        [{'beta': 1.0}],
        [{'label_num': 3}],
        [{'ignore_label': 0}],
        [{'has_ignore_label': True}],
        [{'label_dtype': numpy.int8},
         {'label_dtype': numpy.int16},
         {'label_dtype': numpy.int32},
         {'label_dtype': numpy.int64}]
    )
))
class TestClassificationSummary(unittest.TestCase):

    def setUp(self):
        t_upper = 3 if self.label_num is None else self.label_num
        self.y = numpy.random.uniform(-1, 1, self.y_shape).astype(self.dtype)
        self.t = make_ground_truth(t_upper, self.t_shape,
                                   self.ignore_label, self.has_ignore_label,
                                   self.label_dtype)
        self.check_forward_options = {}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}

    def check_forward(self, xp):
        y = chainer.Variable(xp.asarray(self.y))
        t = chainer.Variable(xp.asarray(self.t))

        pred = self.y.argmax(axis=1).reshape(self.t.shape)
        with numpy.errstate(invalid='ignore'):
            p_expect = precision(pred, self.t, self.dtype,
                                 3, self.ignore_label)
            r_expect = recall(pred, self.t, self.dtype,
                              3, self.ignore_label)
            fbeta_expect = fbeta_score(p_expect, r_expect, self.beta)
            s_expect = support(self.t, self.dtype,
                               3, self.ignore_label)

        # The resultants can include NaN values depending of the inputs.
        # In such case, temporarily disable debug mode to avoid NaN error.
        # TODO(niboshi): separate test cases which can cause NaN and remove
        #   this trick.
        include_nan = (numpy.isnan(p_expect).any()
                       or numpy.isnan(r_expect).any()
                       or numpy.isnan(fbeta_expect).any())

        def forward():
            return F.classification_summary(
                y, t, self.label_num, self.beta, self.ignore_label)

        if include_nan:
            with chainer.using_config('debug', False), \
                    numpy.errstate(invalid='ignore'):
                outputs = forward()
        else:
            outputs = forward()

        p_actual, r_actual, fbeta_actual, s_actual = outputs

        chainer.testing.assert_allclose(p_actual.data, p_expect,
                                        **self.check_forward_options)
        chainer.testing.assert_allclose(r_actual.data, r_expect,
                                        **self.check_forward_options)
        chainer.testing.assert_allclose(fbeta_actual.data, fbeta_expect,
                                        **self.check_forward_options)
        chainer.testing.assert_allclose(s_actual.data, s_expect,
                                        **self.check_forward_options)

    def test_forward_cpu(self):
        self.check_forward(numpy)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.cupy)


testing.run_module(__name__, __file__)
