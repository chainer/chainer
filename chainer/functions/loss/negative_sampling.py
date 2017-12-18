import numpy
import six

import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer.utils import type_check


def _sigmoid_grad(x, y, gy):
    return chainer.functions.activation.sigmoid.SigmoidGrad((x,)).apply(
        (y, gy))[0]


class NegativeSamplingFunction(function_node.FunctionNode):

    ignore_label = -1

    def __init__(self, sampler, sample_size, reduce='sum'):
        if reduce not in ('sum', 'no'):
            raise ValueError(
                "only 'sum' and 'no' are valid for 'reduce', but '%s' is "
                'given' % reduce)

        self.sampler = sampler
        self.sample_size = sample_size
        self.reduce = reduce
        self.wx = None

    def _make_samples(self, t):
        if hasattr(self, 'samples'):
            return self.samples  # for testing

        size = int(t.shape[0])
        # first one is the positive, and others are sampled negatives
        samples = self.sampler((size, self.sample_size + 1))
        samples[:, 0] = t
        self.samples = samples

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        x_type, t_type, w_type = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
            x_type.ndim == 2,
            t_type.dtype == numpy.int32,
            t_type.ndim == 1,
            x_type.shape[0] == t_type.shape[0],
            w_type.dtype == numpy.float32,
            w_type.ndim == 2,
        )

    def forward_cpu(self, inputs):
        self.retain_inputs((0, 1, 2))
        x, t, W = inputs

        self.ignore_mask = (t != self.ignore_label)
        self._make_samples(t)

        w = W[self.samples]
        wx = numpy.einsum(
            'ij,ikj->ik', x[self.ignore_mask], w[self.ignore_mask])
        wx[:, 0] *= -1

        loss = numpy.zeros(len(x), numpy.float32)
        loss[self.ignore_mask] = numpy.sum(numpy.logaddexp(wx, 0), axis=1)

        if self.reduce == 'sum':
            loss = numpy.array(loss.sum(), 'f')
        return loss,

    def forward_gpu(self, inputs):
        self.retain_inputs((0, 1, 2))
        x, t, W = inputs

        self.ignore_mask = (t != self.ignore_label)
        self._make_samples(t)

        n_in = x.shape[1]
        self.wx = cuda.elementwise(
            'raw T W, raw T x, bool mask, S k, int32 c, int32 m', 'T wx',
            '''
            T f = 0;
            if (mask == 1) {
                for (int j = 0; j < c; ++j) {
                  int x_ind[] = {(i / m), j};
                  int w_ind[] = {k, j};
                  f += x[x_ind] * W[w_ind];
                }
            }
            wx = f;
            ''',
            'negative_sampling_wx'
        )(W, x, self.ignore_mask[:, None], self.samples, n_in,
          self.sample_size + 1)

        loss = cuda.elementwise(
            'T wx, int32 c, int32 m, bool mask', 'T y',
            '''
            if (mask) {
              T f = wx;
              if (i % m == 0) {
                f = -f;
              }
              if (f < 0) {
                y = __logf(1 + __expf(f));
              } else {
                y = f + __logf(1 + __expf(-f));
              }
            } else {
              y = 0;
            }
            ''',
            'negative_sampling_forward'
        )(self.wx, n_in, self.sample_size + 1, self.ignore_mask[:, None])

        if self.reduce == 'sum':
            loss = loss.sum()
        else:  # 'no':
            loss = loss.sum(axis=1)
        return loss,

    def backward(self, indexes, grad_outputs):
        x, t, W = self.get_retained_inputs()
        gy, = grad_outputs
        return NegativeSamplingFunctionGrad(
            self.reduce, self.ignore_mask, self.sample_size, self.samples,
            self.wx).apply((x, W, gy))


class NegativeSamplingFunctionGrad(function_node.FunctionNode):

    def __init__(self, reduce, ignore_mask, sample_size, samples, wx):
        self.reduce = reduce
        self.ignore_mask = ignore_mask
        self.sample_size = sample_size
        self.samples = samples
        self.wx = wx

    def forward_cpu(self, inputs):
        self.retain_inputs((0, 1, 2))
        x, W, gloss = inputs

        gx = numpy.zeros_like(x)
        gW = numpy.zeros_like(W)

        for i in numpy.arange(len(self.ignore_mask))[self.ignore_mask]:
            ix = x[i]

            k = self.samples[i]
            if self.reduce == 'sum':
                igy = gloss
            else:
                igy = gloss[i]

            w = W[k]
            f = w.dot(ix)

            # g == -y * gloss / (1 + exp(yf))
            f[0] *= -1
            g = igy / (1 + numpy.exp(-f))
            g[0] *= -1

            gx[i] = g.dot(w)
            for ik, ig in six.moves.zip(k, g):
                gW[ik] += ig * ix
        return gx, None, gW

    def forward_gpu(self, inputs):
        self.retain_inputs((0, 1, 2))
        x, W, gy = inputs

        if self.reduce == 'no':
            gy = gy[:, None]

        g = cuda.elementwise(
            'T wx, T gy, int32 m', 'T g',
            '''
            T y;
            if (i % m == 0) {
              y = 1;
            } else {
              y = -1;
            }

            g = -y * gy / (1.0f + __expf(wx * y));
            ''',
            'negative_sampling_calculate_g'
        )(self.wx, gy, self.sample_size + 1)

        cupy = cuda.cupy
        gx = cupy.zeros_like(x)
        n_in = x.shape[1]
        cuda.elementwise(
            'raw T g, raw T W, bool mask, raw S k, int32 c, int32 m', 'T gx',
            '''
            int d = i / c;
            T w = 0;
            if (mask == 1){
                for (int j = 0; j < m; ++j) {
                  w += g[d * m + j] * W[k[d * m + j] * c + i % c];
                }
            }
            gx = w;
            ''',
            'negative_sampling_calculate_gx'
        )(g, W, self.ignore_mask[:, None], self.samples, n_in,
          self.sample_size + 1, gx)

        gW = cupy.zeros_like(W)
        cuda.elementwise(
            'T g, raw T x, S k, bool mask, int32 c, int32 m',
            'raw T gW',
            '''
            T gi = g;
            if (mask == 1) {
                for (int j = 0; j < c; ++j) {
                  atomicAdd(&gW[k * c + j], gi * x[(i / m) * c + j]);
                }
            }
            ''',
            'negative_sampling_calculate_gw'
        )(g, x, self.samples, self.ignore_mask[:, None], n_in,
          self.sample_size + 1, gW)
        return gx, None, gW

    def backward(self, indexes, grad_outputs):
        x, W, gy = self.get_retained_inputs()

        xp = cuda.get_array_module(x.data)

        if 0 in indexes:
            gx = chainer.Variable(xp.zeros_like(x.data))
        if 1 in indexes:
            gW = chainer.Variable(xp.zeros_like(W.data))
        if 2 in indexes:
            ggy = chainer.Variable(xp.zeros_like(gy.data))

        ggx, _, ggW = grad_outputs

        pos_neg_mask = xp.ones(self.sample_size + 1)
        pos_neg_mask[0] *= -1

        for i in xp.arange(len(self.ignore_mask))[self.ignore_mask]:
            # Partial forward pass to obtain intermediate `Variable`s
            ix = x[i]
            k = self.samples[i]

            if self.reduce == 'sum':
                igy = gy
            else:
                igy = gy[i]

            w = W[k]
            f = chainer.functions.flatten(
                chainer.functions.matmul(w, ix[:, None])) * pos_neg_mask
            sigf = chainer.functions.sigmoid(f)
            g = chainer.functions.broadcast_to(igy, f.shape) * sigf \
                * pos_neg_mask

            dgW_dg = chainer.functions.flatten(
                chainer.functions.matmul(ggW[k], ix[:, None])) * pos_neg_mask
            dgW_df = chainer.functions.broadcast_to(igy, f.shape) \
                * _sigmoid_grad(f, sigf, dgW_dg) * pos_neg_mask
            dgx_dg = chainer.functions.flatten(
                chainer.functions.matmul(ggx[i][None, :], w, transb=True))
            dgx_df = chainer.functions.broadcast_to(igy, f.shape) \
                * _sigmoid_grad(f, sigf, dgx_dg)

            if 0 in indexes:
                # deriative of gx
                dgx = chainer.functions.matmul(w, dgx_df[:, None], transa=True)

                # derivative of gW
                dgx += chainer.functions.matmul(g[None, :], ggW[k]).T
                dgx += chainer.functions.matmul(
                    w, dgW_df[:, None], transa=True)

                gx = chainer.functions.scatter_add(
                    gx, i, chainer.functions.flatten(dgx))

            if 1 in indexes:
                # deriative of gx
                shape = ggx[i].shape
                for ik, ig, idgx_df in six.moves.zip(k, g, dgx_df):
                    ig = chainer.functions.broadcast_to(ig, shape)
                    idgx_df = chainer.functions.broadcast_to(idgx_df, shape)
                    gW = chainer.functions.scatter_add(
                        gW, ik, ig * ggx[i] + idgx_df * ix)

                # derivative of gW
                gW = chainer.functions.scatter_add(
                    gW, k,
                    chainer.functions.matmul(dgW_df[:, None], ix[None, :]))

            if 2 in indexes:
                dgx_dg *= pos_neg_mask
                dggy = chainer.functions.sum((dgx_dg + dgW_dg) * sigf)
                if self.reduce == 'sum':
                    ggy += dggy
                else:
                    ggy = chainer.functions.scatter_add(ggy, i, dggy)

        ret = []
        if 0 in indexes:
            ret.append(gx)
        if 1 in indexes:
            ret.append(gW)
        if 2 in indexes:
            ret.append(ggy)
        return ret


def negative_sampling(x, t, W, sampler, sample_size, reduce='sum'):
    """Negative sampling loss function.

    In natural language processing, especially language modeling, the number of
    words in a vocabulary can be very large.
    Therefore, you need to spend a lot of time calculating the gradient of the
    embedding matrix.

    By using the negative sampling trick you only need to calculate the
    gradient for a few sampled negative examples.

    The loss is defined as follows.

    .. math::

       f(x, p) = - \\log \\sigma(x^\\top w_p) - \\
       k E_{i \\sim P(i)}[\\log \\sigma(- x^\\top w_i)]

    where :math:`\\sigma(\\cdot)` is a sigmoid function, :math:`w_i` is the
    weight vector for the word :math:`i`, and :math:`p` is a positive example.
    It is approximated with :math:`k` examples :math:`N` sampled from
    probability :math:`P(i)`.

    .. math::

       f(x, p) \\approx - \\log \\sigma(x^\\top w_p) - \\
       \\sum_{n \\in N} \\log \\sigma(-x^\\top w_n)

    Each sample of :math:`N` is drawn from the word distribution
    :math:`P(w) = \\frac{1}{Z} c(w)^\\alpha`, where :math:`c(w)` is the
    unigram count of the word :math:`w`, :math:`\\alpha` is a hyper-parameter,
    and :math:`Z` is the normalization constant.

    Args:
        x (~chainer.Variable): Batch of input vectors.
        t (~chainer.Variable): Vector of ground truth labels.
        W (~chainer.Variable): Weight matrix.
        sampler (~types.FunctionType): Sampling function. It takes a shape and
            returns an integer array of the shape. Each element of this array
            is a sample from the word distribution.
            A :class:`~chainer.utils.WalkerAlias` object built with the power
            distribution of word frequency is recommended.
        sample_size (int): Number of samples.
        reduce (str): Reduction option. Its value must be either
            ``'sum'`` or ``'no'``. Otherwise, :class:`ValueError` is raised.

    Returns:
        ~chainer.Variable:
            A variable holding the loss value(s) calculated by the
            above equation.
            If ``reduce`` is ``'no'``, the output variable holds array
            whose shape is same as one of (hence both of) input variables.
            If it is ``'sum'``, the output variable holds a scalar value.

    See: `Distributed Representations of Words and Phrases and their\
         Compositionality <https://arxiv.org/abs/1310.4546>`_

    .. seealso:: :class:`~chainer.links.NegativeSampling`.

    """
    return (
        NegativeSamplingFunction(sampler, sample_size, reduce)
        .apply((x, t, W))
    )[0]
