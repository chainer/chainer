import collections
import numpy
import six

import chainer
from chainer.backends import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check


def _logsumexp(a, xp, axis=None):
    vmax = xp.amax(a, axis=axis, keepdims=True)
    if xp is numpy:
        vmax += xp.log(xp.sum(xp.exp(a - vmax),
                              axis=axis, keepdims=True, dtype=a.dtype))
    else:
        _logsumexp_impl = cuda.reduce(
            'T x, T vmax', 'T y',
            'exp(x - vmax)', 'a + b', 'y += log(a)', '0',
            'logsumexp_impl')
        _logsumexp_impl(a, vmax, vmax, axis=axis, keepdims=True)
    return xp.squeeze(vmax, axis=axis)


def _softmax(x, xp):
    val = xp.exp(x - xp.amax(x, axis=2, keepdims=True))
    val /= xp.sum(val, axis=2, keepdims=True)
    return val


def _label_to_path(labels, blank_symbol, xp):
    path = xp.full((len(labels), labels.shape[1] * 2 + 1),
                   blank_symbol, dtype=numpy.int32)
    path[:, 1::2] = labels
    return path


def _move_label_to_back(path, path_length, xp):
    s1 = path.shape[1]  # TODO(okuta): Change name
    index = (xp.arange(0, path.size, s1, dtype=numpy.int32)[:, None] +
             (xp.arange(s1) + path_length[:, None])[:, ::-1] % s1)
    return xp.take(path, index)


def _move_inputs(prob, input_length, xp):
    seq, batch, ch = prob.shape
    rotate = (xp.arange(seq)[:, None] + input_length) % seq
    index = rotate * batch + xp.arange(batch)
    return xp.take(prob.reshape(seq * batch, ch), index, axis=0)


class ConnectionistTemporalClassification(function.Function):

    """The implementation of Connectionist Temporal Classfication loss functions.

    To make it usable for real-world cases, this class has two policies below.
    1. This class computes forward and backward variables in the log domain.
    2. This class applies the softmax function to inputs. The Backward
    values of CTC loss is often overflows. This is avoided by computing
    backward values before the activation function is applied.
    """

    def __init__(self, blank_symbol, reduce='mean'):
        self.blank_symbol = blank_symbol
        self.zero_padding = -10000000000.0

        if reduce not in ('mean', 'no'):
            raise ValueError(
                "only 'mean' and 'no' are valid "
                "for 'reduce', but '%s' is given" % reduce)
        self.reduce = reduce

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() > 3)  # TODO(okuta): > 3?
        l_type = in_types[2]
        type_check.expect(l_type.dtype == numpy.int32)

        x_basetype = in_types[3]  # TODO(oktua): Check x_basetype size

        for i in six.moves.range(3, len(in_types)):
            x_type = in_types[i]
            type_check.expect(
                x_type.dtype == numpy.float32,
                x_type.shape == x_basetype.shape,
            )

    def log_matrix(self, x, xp):
        if xp == numpy:
            res = numpy.ma.log(x).filled(fill_value=self.zero_padding)
        else:
            create_recurrence_relation = cuda.cupy.ElementwiseKernel(
                'T x, T e', 'T y',
                'y = x == 0 ? e : log(x)',
                'create_recurrence_relation')
            res = create_recurrence_relation(x, self.zero_padding)
        return res.astype(numpy.float32)

    # path probablity to label probability
    def label_probability(self, label_size, path, path_length,
                          multiply_seq, xp):
        seq_length = len(multiply_seq)
        n_batch = len(path)
        dtype = multiply_seq.dtype

        ret = xp.zeros((seq_length, n_batch, label_size), dtype)
        if xp == numpy:
            for b in six.moves.range(len(path)):
                target_path = path[b, :path_length[b]]
                chars = {c for c in target_path}
                for c in chars:
                    ret[:, b, c] = xp.sum(
                        multiply_seq[:, b, 0:path_length[b]]
                        [:, target_path == c], axis=1)
        else:
            cuda.cupy.ElementwiseKernel(
                'T prob, I path, I path_length, I max_path_length',
                'raw T cum_prob',
                '''
                I t = i % max_path_length;
                if (t < path_length) {
                  int n_batch = cum_prob.shape()[1];
                  I s = i / (max_path_length * n_batch);
                  I b = (i - s * (max_path_length * n_batch))
                      / max_path_length;
                  int ind[] = {s, b, path};
                  atomicAdd(&cum_prob[ind], prob);
                }
                ''', 'ctc_label_prob_sum'
            )(multiply_seq, path, path_length[:, None], path.shape[1], ret)
        return ret

    def _computes_transition(self, prev_prob, path, path_length):
        xp = cuda.get_array_module(prev_prob)

        if xp == numpy:
            n_batch, max_path_length = path.shape
            mat = xp.full(
                (3, n_batch, max_path_length), self.zero_padding, 'f')
            mat[0, :, :] = prev_prob
            mat[1, :, 1:] = prev_prob[:, :-1]
            mat[2, :, 2:] = prev_prob[:, :-2]
            # disable transition between the same symbols
            # (including blank-to-blank)
            same_transition = (path[:, :-2] == path[:, 2:])
            mat[2, :, 2:][same_transition] = self.zero_padding
            prob = _logsumexp(mat, xp, axis=0)
            outside = xp.arange(max_path_length) >= path_length[:, None]
            prob[outside] = self.zero_padding
        else:
            prev_prob, path, path_length = xp.broadcast_arrays(
                prev_prob, path, path_length[:, None])
            prob = cuda.elementwise(
                'raw T prob, raw I path, I path_length, T zero', 'T z',
                '''
                int length = prob.shape()[1];
                int b = i / length;
                int t = i - b * length;
                if (t >= path_length) {
                  z = zero;
                  return;
                }
                int ind1[] = {b, t};
                int ind2[] = {b, t - 1};
                int ind3[] = {b, t - 2};
                float f1 = prob[ind1];
                float f2 = (0 <= t - 1) ? prob[ind2] : zero;
                float f3 = (0 <= t - 2 && path[ind3] != path[ind1]) ?
                  prob[ind3] : zero;

                // calculates log-sum-exp
                float m = max(f1, max(f2, f3));
                z = m + log(exp(f1 - m) + exp(f2 - m) + exp(f3 - m));
                ''', 'ctc_transition'
            )(prev_prob, path, path_length, self.zero_padding)
        return prob

    def calc_trans(self, yseq, input_length,
                   label, label_length, path, path_length, xp):
        max_input_length, n_batch, n_unit = yseq.shape
        max_label_length = label.shape[1]
        max_path_length = path.shape[1]
        assert label.shape == (n_batch, max_label_length), label.shape
        assert path.shape == (n_batch, max_label_length * 2 + 1)

        forward_prob = self.log_matrix(
            xp.eye(1, max_path_length, dtype='f'), xp)
        backward_prob = forward_prob
        offset = xp.arange(
            0, n_batch * n_unit, n_unit, dtype=path.dtype)[:, None]

        # prob[i] := forward[i] + backward[-i-1]
        index = offset + path
        prob = xp.empty(
            (max_input_length, n_batch, max_path_length), dtype='f')
        # forward computation.
        for i, y in enumerate(yseq):
            # calc forward probability in log scale
            forward_prob = self._computes_transition(
                forward_prob, path, path_length)
            forward_prob += xp.take(y, index)
            prob[i] = forward_prob

        r_path = _move_label_to_back(path, path_length, xp)
        r_index = offset + r_path

        # rotate yseq with path_length
        yseq_inv = _move_inputs(yseq, input_length, xp)[::-1]
        # move to back.
        prob = _move_inputs(prob, input_length, xp)

        # backward computation.
        backward_prob_index = (
            xp.arange(0, path.size, max_path_length, dtype='i')[:, None] +
            (xp.arange(max_path_length) - path_length[:, None])
            % max_path_length)

        for i, y_inv in enumerate(yseq_inv):
            # calc backward probability
            backward_prob = self._computes_transition(
                backward_prob, r_path, path_length)
            prob[-i - 1] += xp.take(
                backward_prob[:, ::-1], backward_prob_index)
            backward_prob += xp.take(y_inv, r_index)

        # move to front.
        return _move_inputs(prob, -self.input_length, xp)

    def forward(self, inputs):
        xp = cuda.get_array_module(inputs[0])
        self.input_length = inputs[0]
        label_length = inputs[1]
        t = inputs[2]
        xs = inputs[3:]

        if chainer.is_debug():
            # Batch size check.
            assert len(xs[0]) == len(t)
            assert len(xs[0]) == len(self.input_length)
            assert len(xs[0]) == len(label_length)

            # Length check.
            assert len(xs) >= xp.max(self.input_length)
            assert len(t[0]) >= xp.max(label_length)

        self.path_length = 2 * label_length + 1

        yseq_shape = (len(xs),) + xs[0].shape
        self.yseq = _softmax(xp.vstack(xs).reshape(yseq_shape), xp)
        log_yseq = self.log_matrix(self.yseq, xp)
        self.path = _label_to_path(t, self.blank_symbol, xp)
        self.prob_trans = self.calc_trans(
            log_yseq, self.input_length, t,
            label_length, self.path, self.path_length, xp)

        loss = -_logsumexp(self.prob_trans[0], xp, axis=1)
        if self.reduce == 'mean':
            loss = utils.force_array(xp.mean(loss))
        return loss,

    def backward(self, inputs, grad_output):
        xp = cuda.get_array_module(inputs[0])
        batch_size = len(inputs[2])

        total_probability = _logsumexp(self.prob_trans[0], xp, axis=1)
        label_prob = self.label_probability(
            self.yseq.shape[2], self.path, self.path_length,
            xp.exp(self.prob_trans - total_probability[:, None]), xp)
        self.yseq -= label_prob
        if self.reduce == 'mean':
            self.yseq *= grad_output[0] / batch_size
        else:
            self.yseq *= grad_output[0][..., None]
        # mask
        self.yseq *= (
            xp.arange(len(self.yseq))[:, None] < self.input_length)[..., None]
        return (None, None, None) + tuple([y for y in self.yseq])


def connectionist_temporal_classification(
        x, t, blank_symbol, input_length=None, label_length=None,
        reduce='mean'):
    """Connectionist Temporal Classification loss function.

    Connectionist Temporal Classification(CTC) [Graves2006]_ is a loss function
    of sequence labeling where the alignment between the inputs and target is
    unknown. See also [Graves2012]_

    The output is a variable whose value depends on the value of
    the option ``reduce``. If it is ``'no'``, it holds the samplewise
    loss values. If it is ``'mean'``, it takes the mean of loss values.


    Args:
        x (list or tuple of :class:`~chainer.Variable`):
            RNN output at each time. Each element of ``x``, ``x[i]``
            is a :class:`~chainer.Variable` representing output of RNN at time
            ``i``.
        t (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Expected label sequence.
        blank_symbol (int): Index of blank_symbol.
            This value must be non-negative.
        input_length (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Length of valid sequence for each of mini
            batch ``x`` (optional). If input_length is skipped, It regards that
            all of ``x`` is valid input.
        label_length (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Length of valid sequence for each of mini
            batch ``t`` (optional). If label_length is skipped, It regards that
            all of ``t`` is valid input.
        reduce (str): Reduction option. Its value must be either
            ``'mean'`` or ``'no'``. Otherwise,
            :class:`ValueError` is raised.

    Returns:
       ~chainer.Variable:
           A variable holding a scalar value of the CTC loss.
           If ``reduce`` is ``'no'``, the output variable holds array
           whose shape is `(B,)` where `B` is the number of samples.
           If it is ``'mean'``, it holds a scalar.

    .. note::
       You need to input ``x`` without applying to activation functions(e.g.
       softmax function), because this function applies softmax functions
       to ``x`` before calculating CTC loss to avoid numerical limitations.
       You also need to apply softmax function to forwarded values before you
       decode it.

    .. note::
       This function is differentiable only by ``x``.

    .. note::
       This function supports (batch, sequence, 1-dimensional input)-data.

    .. [Graves2006] Alex Graves, Santiago Fernandez,\
    Faustino Gomez, Jurgen Schmidhuber,\
    `Connectionist Temporal Classification: Labelling Unsegmented\
    Sequence Data with Recurrent Neural Networks\
    <ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf>`_

    .. [Graves2012] Alex Graves,\
    `Supervised Sequence Labelling with Recurrent Neural Networks\
    <https://www.cs.toronto.edu/~graves/preprint.pdf>`_

    """
    if not isinstance(x, collections.Sequence):
        raise TypeError('x must be a list of Variables')
    if not isinstance(blank_symbol, int):
        raise TypeError('blank_symbol must be non-negative integer.')
    assert 0 <= blank_symbol < x[0].shape[1]
    # This implementation only supports 1-dimensional data.
    # TODO(jnishi): Support d(>1)-dimentinal inputs.
    assert x[0].ndim == 2

    xp = cuda.get_array_module(x[0])
    if input_length is None:
        input_length = xp.full(len(x[0]), len(x), dtype=numpy.int32)
    if label_length is None:
        label_length = xp.full(len(t), t.shape[1], dtype=numpy.int32)

    return ConnectionistTemporalClassification(blank_symbol, reduce)(
        input_length, label_length, t, *x)
