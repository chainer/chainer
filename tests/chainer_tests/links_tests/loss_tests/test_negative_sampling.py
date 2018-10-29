import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import links
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    't': [[0, 2], [-1, 1, 2]],
    'reduce': ['sum', 'no'],
}))
class TestNegativeSampling(unittest.TestCase):

    in_size = 3
    sample_size = 2

    def setUp(self):
        batch = len(self.t)
        x_shape = (batch, self.in_size)
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(numpy.float32)
        self.t = numpy.array(self.t).astype(numpy.int32)

        if self.reduce == 'no':
            g_shape = self.t.shape
        elif self.reduce == 'sum':
            g_shape = ()
        self.gy = numpy.random.uniform(-1, 1, g_shape).astype(numpy.float32)

    def create_link(self, rng=None):
        if rng is None:
            rng = numpy.random.RandomState()
        link = links.NegativeSampling(
            self.in_size, [10, 5, 2, 5, 2], self.sample_size)
        link.cleargrads()
        # W is initialized with zero. Inject random values for meaningful test.
        link.W.array[:] = rng.uniform(-1, 1, link.W.shape)
        return link

    def call_link_and_return_samples(self, func):
        # Calls the link while sneaking the samples returned from
        # F.negative_sampling.
        # func is a function in which the link is called.

        # Wrap F.negative_sampling to sneak the samples.
        orig_negative_sampling = chainer.functions.negative_sampling
        saved_samples = [None]

        def wrap_negative_sampling(*args, **kwargs):
            out, samples = orig_negative_sampling(
                *args, return_samples=True, **kwargs)
            saved_samples[0] = samples
            return out

        with testing.patch(
                'chainer.functions.loss.negative_sampling.negative_sampling',
                wraps=wrap_negative_sampling) as m:
            out = func()
            assert m.call_count == 1

        assert saved_samples[0] is not None
        return out, saved_samples[0]

    def check_forward(self, link, x_data, t_data):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        y, samples = self.call_link_and_return_samples(
            lambda: link(x, t, reduce=self.reduce))
        self.assertEqual(y.shape, self.gy.shape)

        W = cuda.to_cpu(link.W.data)
        samples = cuda.to_cpu(samples)

        loss = numpy.empty((len(self.x),), numpy.float32)
        for i in range(len(self.x)):
            ix = self.x[i]
            it = self.t[i]
            if it == -1:
                loss[i] = 0
            else:
                w = W[samples[i]]
                f = w.dot(ix)
                # first one is positive example
                f[0] *= -1
                loss[i] = numpy.logaddexp(f, 0).sum()

        if self.reduce == 'sum':
            loss = loss.sum()

        testing.assert_allclose(y.data, loss)

    def test_forward_cpu(self):
        link = self.create_link()
        self.check_forward(link, self.x, self.t)

    @attr.gpu
    def test_forward_gpu(self):
        link = self.create_link()
        link.to_gpu()
        self.check_forward(link, cuda.to_gpu(self.x), cuda.to_gpu(self.t))

    @attr.gpu
    def test_to_cpu(self):
        link = self.create_link()
        link.to_gpu()
        self.assertTrue(link.sampler.use_gpu)
        link.to_cpu()
        self.assertFalse(link.sampler.use_gpu)

    @attr.gpu
    def test_backward_cpu_gpu(self):
        # This test compares gradients of CPU and GPU modes.

        rng = numpy.random.RandomState()
        rng_state = rng.get_state()

        # Call CPU mode link and save samples
        x = chainer.Variable(self.x)
        t = chainer.Variable(self.t)
        link = self.create_link(rng)

        y, samples = self.call_link_and_return_samples(
            lambda: link(x, t))

        y.backward()
        assert t.grad is None
        gw_cpu = link.W.grad
        gx_cpu = x.grad

        # mock sampler that returns the saved samples from CPU mode
        def mock_sample_gpu(shape):
            assert samples.shape == shape
            return cuda.to_gpu(samples)

        # Wrap F.negative_sampling to replace sampler with the mock
        orig_negative_sampling = chainer.functions.negative_sampling

        def wrap_negative_sampling2(*args, **kwargs):
            args = args[:3] + (mock_sample_gpu,) + args[4:]
            return orig_negative_sampling(*args, **kwargs)

        # Call GPU mode link
        rng.set_state(rng_state)
        link = self.create_link(rng)
        link.to_gpu()
        x = chainer.Variable(cuda.to_gpu(self.x))
        t = chainer.Variable(cuda.to_gpu(self.t))
        with testing.patch(
                'chainer.functions.loss.negative_sampling.negative_sampling',
                wraps=wrap_negative_sampling2) as m:
            y = link(x, t)
            assert m.call_count == 1

        y.backward()
        assert t.grad is None
        gw_gpu = link.W.grad
        gx_gpu = x.grad

        # Compare gradients from CPU and GPU modes
        testing.assert_allclose(gx_cpu, gx_gpu, atol=1.e-4)
        testing.assert_allclose(gw_cpu, gw_gpu, atol=1.e-4)


testing.run_module(__name__, __file__)
