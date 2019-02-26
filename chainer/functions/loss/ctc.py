import numpy
import six

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import function
from chainer import utils
from chainer.utils import collections_abc
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


def _flip_path(path, path_length, xp):
    """Flips label sequence.

    This function rotates a label sequence and flips it.
    ``path[b, t]`` stores a label at time ``t`` in ``b``-th batch.
    The rotated matrix ``r`` is defined as
    ``r[b, t] = path[b, t + path_length[b]]``

    .. ::

       a b c d .     . a b c d    d c b a .
       e f . . .  -> . . . e f -> f e . . .
       g h i j k     g h i j k    k j i h g

    """
    n_batch, n_label = path.shape
    rotate = (xp.arange(n_label) + path_length[:, None]) % n_label
    return path[xp.arange(n_batch, dtype=xp.int32)[:, None],
                rotate][:, ::-1]


def _flip_label_probability(y, input_length, xp):
    """Flips a label probability matrix.

    This function rotates a label probability matrix and flips it.
    ``y[i, b, l]`` stores log probability of label ``l`` at ``i``-th
    input in ``b``-th batch.
    The rotated matrix ``r`` is defined as
    ``r[i, b, l] = y[i + input_length[b], b, l]``

    """
    seq, n_batch, n_vocab = y.shape
    rotate = (xp.arange(seq, dtype=xp.int32)[:, None] + input_length) % seq
    return y[
        rotate[:, :, None],
        xp.arange(n_batch, dtype=xp.int32)[None, :, None],
        xp.arange(n_vocab, dtype=xp.int32)[None, None, :]][::-1]


def _flip_path_probability(prob, input_length, path_length, xp):
    """Flips a path probability matrix.

    This function returns a path probability matrix and flips it.
    ``prob[i, b, t]`` stores log probability at ``i``-th input and
    at time ``t`` in a output sequence in ``b``-th batch.
    The rotated matrix ``r`` is defined as
    ``r[i, j, k] = prob[i + input_length[j], j, k + path_length[j]]``

    """
    seq, n_batch, n_label = prob.shape
    rotate_input = ((xp.arange(seq, dtype=xp.int32)[:, None] + input_length)
                    % seq)
    rotate_label = ((xp.arange(n_label, dtype=xp.int32) + path_length[:, None])
                    % n_label)
    return prob[
        rotate_input[:, :, None],
        xp.arange(n_batch, dtype=xp.int32)[None, :, None],
        rotate_label][::-1, :, ::-1]


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
        # Lazily initialized in the first forward computation for dtype
        self.zero_padding = None

        if reduce not in ('mean', 'no'):
            raise ValueError(
                'only \'mean\' and \'no\' are valid '
                'for \'reduce\', but \'%s\' is given' % reduce)
        self.reduce = reduce

    def check_type_forward(self, in_types):
        type_check._argname(
            in_types, ('input_length', 'label_length', 't', 'x'))
        input_length_type, label_length_type, t_type, x_type = in_types
        type_check.expect(
            input_length_type.dtype == numpy.int32,
            input_length_type.ndim == 1,
            label_length_type.dtype == numpy.int32,
            label_length_type.ndim == 1,
            t_type.ndim == 2,
            t_type.dtype == numpy.int32,
            x_type.ndim == 3,
            x_type.dtype.kind == 'f',
        )
        n_batch = x_type.shape[1]
        type_check.expect(
            t_type.shape[0] == n_batch,
            input_length_type.shape[0] == n_batch,
            label_length_type.shape[0] == n_batch,
        )

    def log_matrix(self, x, xp):
        if xp == numpy:
            res = numpy.ma.log(x).filled(fill_value=self.zero_padding)
        else:
            create_recurrence_relation = cuda.elementwise(
                'T x, T e', 'T y',
                'y = x == 0 ? e : (T)log(x)',
                'create_recurrence_relation')
            res = create_recurrence_relation(x, self.zero_padding)
        return res.astype(x.dtype, copy=False)

    # path probability to label probability
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
            utils.nondeterministic('atomicAdd')
            cuda.elementwise(
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

    def _computes_transition(
            self, prev_prob, path, path_length, cum_prob, y):
        xp = backend.get_array_module(prev_prob)

        if xp == numpy:
            n_batch, max_path_length = path.shape
            mat = xp.full(
                (3, n_batch, max_path_length), self.zero_padding, y.dtype)
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
            cum_prob += prob
            batch_index = xp.arange(n_batch, dtype=xp.int32)
            prob += y[batch_index[:, None], path]
        else:
            prob = xp.empty_like(prev_prob)
            cuda.elementwise(
                'raw T prob, raw I path, I path_length, T zero, raw T y',
                'T z, T cum_prob',
                '''
                int length = prob.shape()[1];
                int b = i / length;
                int t = i - b * length;
                if (t >= path_length) {
                  z = zero;
                  cum_prob += zero;
                  return;
                }
                int ind1[] = {b, t};
                int ind2[] = {b, t - 1};
                int ind3[] = {b, t - 2};
                T f1 = prob[ind1];
                T f2 = (0 <= t - 1) ? prob[ind2] : zero;
                T f3 = (0 <= t - 2 && path[ind3] != path[ind1]) ?
                  prob[ind3] : zero;

                // calculates log-sum-exp
                T m = max(f1, max(f2, f3));
                z = m + log(exp(f1 - m) + exp(f2 - m) + exp(f3 - m));

                cum_prob += z;

                int y_ind[] = {b, path[ind1]};
                z += y[y_ind];
                ''', 'ctc_transition'
            )(prev_prob, path, path_length[:, None], self.zero_padding, y,
              prob, cum_prob)
        return prob

    def calc_trans(self, yseq, input_length,
                   label, label_length, path, path_length, xp):
        max_input_length, n_batch, n_unit = yseq.shape
        max_label_length = label.shape[1]
        max_path_length = path.shape[1]
        assert label.shape == (n_batch, max_label_length), label.shape
        assert path.shape == (n_batch, max_label_length * 2 + 1)

        forward_prob = xp.full(
            (n_batch, max_path_length), self.zero_padding, dtype=yseq.dtype)
        forward_prob[:, 0] = 0
        backward_prob = forward_prob

        batch_index = xp.arange(n_batch, dtype=xp.int32)
        seq_index = xp.arange(len(yseq), dtype=xp.int32)
        prob = yseq[seq_index[:, None, None], batch_index[:, None], path]
        # forward computation.
        for i, y in enumerate(yseq):
            forward_prob = self._computes_transition(
                forward_prob, path, path_length, prob[i], y)

        r_path = _flip_path(path, path_length, xp)

        yseq_inv = _flip_label_probability(yseq, input_length, xp)
        prob = _flip_path_probability(prob, input_length, path_length, xp)

        for i, y_inv in enumerate(yseq_inv):
            backward_prob = self._computes_transition(
                backward_prob, r_path, path_length, prob[i], y_inv)

        return _flip_path_probability(prob, input_length, path_length, xp)

    def forward(self, inputs):
        xp = backend.get_array_module(inputs[0])
        self.input_length, label_length, t, xs = inputs

        if self.zero_padding is None:
            if xs.dtype == numpy.float16:
                self.zero_padding = -10000.0
            else:
                self.zero_padding = -10000000000.0

        if chainer.is_debug():
            assert len(xs) >= xp.max(self.input_length)
            assert t.shape[1] >= xp.max(label_length)

        self.path_length = 2 * label_length + 1

        self.yseq = _softmax(xs, xp)
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
        xp = backend.get_array_module(inputs[0])
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
        return None, None, None, self.yseq


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
            A list of unnormalized probabilities for labels.
            Each element of ``x``, ``x[i]`` is a :class:`~chainer.Variable`
            object, which has shape ``(B, V)``, where ``B``
            is the batch size and ``V`` is the number of labels.
            The softmax of ``x[i]`` represents the probabilities of the labels
            at time ``i``.
        t (:class:`~chainer.Variable` or :ref:`ndarray`):
            A matrix including expected label sequences.
            Its shape is ``(B, M)``, where ``B`` is the batch size and ``M`` is
            the maximum length of the label sequences.
            All elements in ``t`` must be less than ``V``, the number of
            labels.
        blank_symbol (int): Index of blank_symbol.
            This value must be non-negative.
        input_length (:class:`~chainer.Variable` or :ref:`ndarray`):
            Length of sequence for each of mini batch ``x`` (optional).
            Its shape must be ``(B,)``.
            If the ``input_length`` is omitted or ``None``, it assumes that
            all of ``x`` is valid input.
        label_length (:class:`~chainer.Variable` or :ref:`ndarray`):
            Length of sequence for each of mini batch ``t`` (optional).
            Its shape must be ``(B,)``.
            If the ``label_length`` is omitted or ``None``, it assumes that
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
    if not isinstance(x, collections_abc.Sequence):
        raise TypeError('x must be a list of Variables')
    if not isinstance(blank_symbol, int):
        raise TypeError('blank_symbol must be non-negative integer.')
    assert 0 <= blank_symbol < x[0].shape[1]
    # This implementation only supports 1-dimensional data.
    # TODO(jnishi): Support d(>1)-dimensional inputs.
    assert x[0].ndim == 2

    xp = backend.get_array_module(x[0])
    if input_length is None:
        input_length = xp.full(len(x[0]), len(x), dtype=numpy.int32)
    if label_length is None:
        label_length = xp.full(len(t), t.shape[1], dtype=numpy.int32)

    return ConnectionistTemporalClassification(blank_symbol, reduce)(
        input_length, label_length, t, chainer.functions.stack(x))
