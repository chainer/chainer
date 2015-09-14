import numpy
import six

from chainer import cuda
from chainer import function
from chainer import model
from chainer.utils import type_check
from chainer.utils import walker_alias
from chainer import variable


class NegativeSamplingFunction(function.Function):

    """Negative sampling loss function.

    Args:
        counts (int list): Number of each identifiers.
        sample_size (int): Number of negative samples.
        power (float): Power factor :math:`\\alpha`.

    .. seealso::
        See :class:`NegativeSampling` for full documentation.

    """
    def __init__(self, counts, sample_size, power=0.75):
        self.sample_size = sample_size
        power = numpy.float32(power)
        p = numpy.array(counts, power.dtype)
        p = numpy.power(p, power)
        self.sampler = walker_alias.WalkerAlias(p)
        self.samples = None
        self.vocab_size = len(counts)

    def _make_samples(self, t):
        if self.samples is not None:
            return self.samples

        size = int(t.shape[0])
        # first one is the positive, and others are sampled negatives
        samples = self.sampler.sample((size, self.sample_size + 1))
        samples[:, 0] = t
        self.samples = samples

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        x_type, t_type, w_type = in_types

        vocab_size = type_check.Variable(self.vocab_size, "len(counts)")
        type_check.expect(
            x_type.dtype == numpy.float32,
            x_type.ndim == 2,
            t_type.dtype == numpy.int32,
            t_type.ndim == 1,
            x_type.shape[0] == t_type.shape[0],
            w_type.dtype == numpy.float32,
            w_type.ndim == 2,
            w_type.shape[0] == vocab_size,
        )

    def to_gpu(self, device=None):
        with cuda.get_device(device):
            self.sampler.to_gpu()
            if self.samples is not None:
                self.samples = cuda.to_gpu(self.samples)

    def to_cpu(self):
        self.sampler.to_cpu()
        if self.samples is not None:
            self.samples = cuda.to_cpu(self.samples)

    def forward_cpu(self, inputs):
        x, t, W = inputs
        self._make_samples(t)

        loss = numpy.float32(0.0)
        for i, (ix, k) in enumerate(six.moves.zip(x, self.samples)):
            w = W[k]
            f = w.dot(ix)
            f[0] *= -1  # positive sample
            loss += numpy.sum(numpy.logaddexp(f, 0))
        return numpy.array(loss, numpy.float32),

    def forward_gpu(self, inputs):
        x, t, W = inputs
        n_in = x.shape[1]
        self._make_samples(t)

        self.wx = cuda.elementwise(
            'raw T W, raw T x, S k, int32 c, int32 m', 'T wx',
            '''
            T f = 0;
            for (int j = 0; j < c; ++j) {
              int x_ind[] = {(i / m), j};
              int w_ind[] = {k, j};
              f += x[x_ind] * W[w_ind];
            }
            wx = f;
            ''',
            'negative_sampling_wx'
        )(W, x, self.samples, n_in, self.sample_size + 1)

        y = cuda.elementwise(
            'T wx, int32 c, int32 m', 'T y',
            '''
            T f = wx;
            if (i % m == 0) {
              f = -f;
            }
            T loss;
            if (f < 0) {
              loss = __logf(1 + __expf(f));
            } else {
              loss = f + __logf(1 + __expf(-f));
            }
            y = loss;
            ''',
            'negative_sampling_forward'
        )(self.wx, n_in, self.sample_size + 1)
        # TODO(okuta): merge elementwise
        loss = cuda.cupy.sum(y)
        return loss,

    def backward_cpu(self, inputs, grads):
        x, t, W = inputs
        gloss, = grads

        gx = numpy.zeros_like(x)
        gW = numpy.zeros_like(W)
        for i, (ix, k) in enumerate(six.moves.zip(x, self.samples)):
            w = W[k]
            f = w.dot(ix)

            # g == -y * gloss / (1 + exp(yf))
            f[0] *= -1
            g = gloss / (1 + numpy.exp(-f))
            g[0] *= -1

            gx[i] = g.dot(w)
            for ik, ig in six.moves.zip(k, g):
                gW[ik] += ig * ix
        return gx, None, gW

    def backward_gpu(self, inputs, grads):
        cupy = cuda.cupy
        x, t, W = inputs
        gloss, = grads

        n_in = x.shape[1]
        g = cuda.elementwise(
            'T wx, raw T gloss, int32 m', 'T g',
            '''
            T y;
            if (i % m == 0) {
              y = 1;
            } else {
              y = -1;
            }

            g = -y * gloss[0] / (1.0f + __expf(wx * y));
            ''',
            'negative_sampling_calculate_g'
        )(self.wx, gloss, self.sample_size + 1)
        gx = cupy.zeros_like(x)
        cuda.elementwise(
            'raw T g, raw T W, raw S k, int32 c, int32 m', 'T gx',
            '''
            int d = i / c;
            T w = 0;
            for (int j = 0; j < m; ++j) {
              w += g[d * m + j] * W[k[d * m + j] * c + i % c];
            }
            gx = w;
            ''',
            'negative_sampling_calculate_gx'
        )(g, W, self.samples, n_in, self.sample_size + 1, gx)
        gW = cupy.zeros_like(W)
        cuda.elementwise(
            'T g, raw T x, S k, int32 c, int32 m', 'raw T gW',
            '''
            T gi = g;
            for (int j = 0; j < c; ++j) {
              atomicAdd(&gW[k * c + j], gi * x[(i / m) * c + j]);
            }
            ''',
            'negative_sampling_calculate_gw'
        )(g, x, self.samples, n_in, self.sample_size + 1, gW)
        return gx, None, gW


class NegativeSampling(model.Model):
    """Model implementation of negative sampling loss.

    In natural language processing, especially language modeling, the number of
    vocabulary is very large.
    Therefore, you need to spend a lot of time to calculate the gradient of the
    embedding matrix.

    Instead, in negative sampling trick, you only need to calculate the
    gradient for a few sampled negative examples.

    The objective function is below:

    .. math::

       f(x, p) = \log\sigma(x^\\top w_p) + \\
       k E_{i \sim P(i)}[\log\sigma(- x^\\top w_i)],

    where :math:`\sigma(\cdot)` is a sigmoid function, :math:`w_i` is the
    weight vector for the word :math:`i`, and :math:`p` is a positive example.
    It is approximeted with :math:`k` examples :math:`N` sampled from
    probability :math:`P(i)`, like this:

    .. math::

       f(x, p) \\approx \log\sigma(x^\\top w_p) + \\
       \sum_{n \in N} \log\sigma(-x^\\top w_n).

    Each sample of :math:`N` is drawn from the word distribution :math:`P(w)`.
    This is calculated as :math:`P(w) = \\frac{1}{Z} c(w)^\\alpha`, where
    :math:`c(w)` is the unigram count of the word :math:`w`, :math:`\\alpha` is
    a hyper-parameter, and :math:`Z` is the normalization constant.

    Args:
        in_size (int): Dimension of input vectors.
        counts (int list): Number of each identifiers.
        sample_size (int): Number of negative samples.
        power (float): Power factor :math:`\\alpha`.

    See: `Distributed Representations of Words and Phrases and their\
         Compositionality <http://arxiv.org/abs/1310.4546>`_

    """
    def __init__(self, in_size, counts, sample_size, power=0.75):
        super(NegativeSampling, self).__init__()
        self._func = NegativeSamplingFunction(counts, sample_size, power)
        vocab_size = len(counts)
        self.params['W'] = variable.Variable(
            numpy.zeros((vocab_size, in_size)).astype(numpy.float32))

    def to_gpu(self, device=None):
        super(NegativeSampling, self).to_gpu(device)
        self._func.to_gpu(device)

    def to_cpu(self):
        super(NegativeSampling, self).to_cpu()
        self._func.to_cpu()

    def __call__(self, x, t):
        return self._func(x, t, self.params['W'])
