import numpy
import six

import chainer
from chainer.functions.normalization import batch_renormalization
from chainer import testing
import chainerx


# naive implementation of differentiable batch renormalization
def _naive_batch_renormalization(
        x, gamma, beta,  # variables
        mean, var,  # variables
        running_mean, running_var,  # arrays
        rmax, dmax, eps, decay):
    # If decay is not None, the running stats are updated.

    F = chainer.functions
    assert isinstance(x, chainer.Variable)
    assert isinstance(gamma, chainer.Variable)
    assert isinstance(beta, chainer.Variable)
    assert isinstance(mean, chainer.Variable)
    assert isinstance(var, chainer.Variable)
    assert isinstance(running_mean, chainer.get_array_types())
    assert isinstance(running_var, chainer.get_array_types())
    assert mean.shape == var.shape
    assert mean.shape == running_mean.shape
    assert mean.shape == running_var.shape
    assert mean.shape == gamma.shape

    dt = x.dtype.type
    std = F.sqrt(var + dt(eps))
    # r and d are gradient-stopped
    running_std = numpy.sqrt(running_var + dt(eps))
    r = (std.array / running_std).clip(1. / rmax, rmax)
    d = ((mean.array - running_mean) / running_std).clip(-dmax, dmax)
    xhat = (x - mean) / std * r + d
    y = gamma * xhat + beta

    # Update running stats
    if decay is not None:
        running_mean *= decay
        running_mean += mean.array * dt(1. - decay)
        # unbiased estimation
        m = x.size // gamma.size
        adjust = m / max(m - 1., 1.)
        running_var *= decay
        running_var += (var.array + dt(eps)) * dt((1. - decay) * adjust)

    return y


def parameterize_batch_renormalization():
    return testing.parameterize(*(testing.product({
        'ndim': [0, 1, 2],
        'eps': [2e-5, 1e-1],
        'dtype': [numpy.float32],
        'update_statistics': [True, False],
    }) + testing.product({
        'ndim': [1],
        'eps': [2e-5, 1e-1],
        'dtype': [numpy.float16, numpy.float32, numpy.float64],
        'update_statistics': [True, False],
    })))


def inject_backend_tests_batch_renormalization():
    return testing.inject_backend_tests(
        None,
        # CPU tests
        [
            {},
            {'use_ideep': 'always'},
        ]
        # GPU tests
        + [
            {'use_cuda': True, 'cuda_device': 0},
            {'use_cuda': True, 'cuda_device': 1},
        ]
        # ChainerX tests
        + [
            {'use_chainerx': True, 'chainerx_device': 'native:0'},
            {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
            {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
        ])


@parameterize_batch_renormalization()
@inject_backend_tests_batch_renormalization()
class TestBatchRenormalizationForward(testing.FunctionTestCase):
    # As F.batch_renormalization includes a calculation in which the outcome
    # depends on x but it's gradient-stopped w.r.t. x,
    # gradient_check.check_backward cannot be used, because numerical
    # gradients would not be calculated correctly.
    # Instead of using gradient_check.check_backward, this test checks the
    # backward gradients as a "forward" function.
    # In addition, updated running_mean and running_var are also included in
    # the outputs of the "forward" function.

    skip_backward_test = True
    skip_double_backward_test = True

    rmax = 3
    dmax = 5

    def setUp(self):
        shape = (5, 3) + (2,) * self.ndim
        aggr_shape = (3,)
        self.running_mean = (
            numpy.random.uniform(-1, 1, aggr_shape).astype(self.dtype))
        self.running_var = (
            numpy.random.uniform(1e-3, 1, aggr_shape).astype(self.dtype))

        axis = (0,) + tuple(six.moves.range(2, self.ndim + 2))
        expander = (None, Ellipsis) + (None,) * self.ndim

        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 0.02, 'rtol': 0.02}

        self.shape = shape
        self.aggr_shape = aggr_shape
        self.axis = axis
        self.expander = expander

    def generate_inputs(self):
        shape = self.shape
        aggr_shape = self.aggr_shape
        dtype = self.dtype

        x = numpy.random.uniform(-10, 10, shape).astype(dtype)
        gamma = numpy.random.uniform(.5, 1, aggr_shape).astype(dtype)
        beta = numpy.random.uniform(-1, 1, aggr_shape).astype(dtype)
        gy = numpy.random.uniform(-1, 1, shape).astype(dtype)
        return x, gamma, beta, gy

    def _compute_backward(self, x, gamma, beta, y, gy):
        assert isinstance(x, chainer.Variable)
        assert isinstance(gamma, chainer.Variable)
        assert isinstance(beta, chainer.Variable)
        assert isinstance(y, chainer.Variable)
        assert isinstance(gy, chainer.Variable)

        if x.xp is chainerx:
            # TODO(niboshi): ChainerX does not support grad yet
            y.grad = gy.array.copy()
            y.backward()
            gx = x.grad_var
            ggamma = gamma.grad_var
            gbeta = beta.grad_var
        else:
            gx, ggamma, gbeta = chainer.grad([y], [x, gamma, beta], [gy])
        return gx.array, ggamma.array, gbeta.array

    def forward(self, inputs, device):
        x, gamma, beta, gy = inputs
        running_mean = device.send(self.running_mean.copy())
        running_var = device.send(self.running_var.copy())
        y = batch_renormalization.batch_renormalization(
            x, gamma, beta,
            self.rmax, self.dmax,
            eps=self.eps,
            running_mean=running_mean,
            running_var=running_var,
            update_statistics=self.update_statistics)

        # backward gradients
        gx, ggamma, gbeta = self._compute_backward(x, gamma, beta, y, gy)

        return (
            y,
            chainer.Variable(running_mean),
            chainer.Variable(running_var),
            chainer.Variable(gx),
            chainer.Variable(ggamma),
            chainer.Variable(gbeta),
        )

    def forward_expected(self, inputs):
        F = chainer.functions
        expander = self.expander
        axis = self.axis

        if self.update_statistics:
            decay = 0.9  # defaut value of F.batch_renormalization
        else:
            decay = None

        x_arr, gamma_arr, beta_arr, gy_arr = inputs
        x = chainer.Variable(x_arr)
        gamma = chainer.Variable(gamma_arr[expander])
        beta = chainer.Variable(beta_arr[expander])

        x_mean = F.mean(x, axis=axis, keepdims=True)
        x_var = F.mean((x - x_mean) ** 2, axis=axis, keepdims=True)
        running_mean = self.running_mean.copy()
        running_var = self.running_var.copy()
        y = _naive_batch_renormalization(
            x, gamma, beta,
            x_mean,
            x_var,
            running_mean[expander],
            running_var[expander],
            self.rmax, self.dmax, self.eps,
            decay)

        # backward gradients
        gx, ggamma, gbeta = self._compute_backward(
            x, gamma, beta, y, chainer.Variable(gy_arr))
        ggamma = numpy.squeeze(ggamma, axis)
        gbeta = numpy.squeeze(gbeta, axis)

        return (
            y.array,
            running_mean, running_var,
            gx, ggamma, gbeta)


@testing.parameterize(*testing.product({
    'ndim': [0, 1, 2, 3],
    'eps': [2e-5, 1e-1],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@inject_backend_tests_batch_renormalization()
class TestFixedBatchRenormalization(testing.FunctionTestCase):

    def setUp(self):
        if self.dtype == numpy.float16:
            self.check_forward_options.update({'atol': 5e-3, 'rtol': 1e-2})
            self.check_backward_options.update({'atol': 5e-3, 'rtol': 1e-2})
            self.check_double_backward_options.update(
                {'atol': 5e-3, 'rtol': 1e-2})

    def generate_inputs(self):
        channels = 3
        shape = (5, channels) + (2,) * self.ndim
        dtype = self.dtype
        x = numpy.random.uniform(-1, 1, shape).astype(dtype)
        gamma = numpy.random.uniform(.5, 1, (channels,)).astype(dtype)
        beta = numpy.random.uniform(-1, 1, (channels,)).astype(dtype)
        mean = numpy.random.uniform(-1, 1, (channels,)).astype(dtype)
        var = numpy.random.uniform(0.5, 1, (channels,)).astype(dtype)
        return x, gamma, beta, mean, var

    def forward(self, inputs, device):
        x, gamma, beta, mean, var = inputs
        with testing.assert_warns(DeprecationWarning):
            y = batch_renormalization.fixed_batch_renormalization(
                x, gamma, beta, mean, var, eps=self.eps)
        return y,

    def forward_expected(self, inputs):
        expander = (None, Ellipsis) + (None,) * self.ndim

        x_arr, gamma_arr, beta_arr, mean_arr, var_arr = inputs
        x = chainer.Variable(x_arr)
        gamma = chainer.Variable(gamma_arr[expander])
        beta = chainer.Variable(beta_arr[expander])

        y = _naive_batch_renormalization(
            x, gamma, beta,
            chainer.Variable(mean_arr[expander]),
            chainer.Variable(var_arr[expander]),
            mean_arr[expander], var_arr[expander],
            1, 0, self.eps, None)
        return y.array,


testing.run_module(__name__, __file__)
