import numpy

from chainer import backend
from chainer import function_node
from chainer.utils import argument
from chainer.utils import type_check


# {numpy: True, cupy: False}
_xp_supports_batch_eigh = {}


# routines for batched matrices
def _eigh(a, xp):
    if xp not in _xp_supports_batch_eigh:
        try:
            xp.linalg.eigh(xp.ones((2, 2, 2), xp.float32))
        except ValueError:
            _xp_supports_batch_eigh[xp] = False
        else:
            _xp_supports_batch_eigh[xp] = True
    if _xp_supports_batch_eigh[xp]:
        return xp.linalg.eigh(a)
    ws = []
    vs = []
    for ai in a:
        w, v = xp.linalg.eigh(ai)
        ws.append(w)
        vs.append(v)
    return xp.stack(ws), xp.stack(vs)


def _matmul(a, b, xp):
    if hasattr(xp, 'matmul'):  # numpy.matmul is supported from version 1.10.0
        return xp.matmul(a, b)
    else:
        return xp.einsum('bij,bjk->bik', a, b)


def _diag(a, xp):
    s0, s1 = a.shape
    ret = xp.zeros((s0, s1, s1), a.dtype)
    arange_s1 = numpy.arange(s1)
    ret[:, arange_s1, arange_s1] = a
    return ret


def _calc_axis_and_m(x_shape, batch_size):
    m = batch_size
    spatial_ndim = len(x_shape) - 2
    spatial_axis = tuple(range(2, 2 + spatial_ndim))
    for i in spatial_axis:
        m *= x_shape[i]
    return spatial_axis, m


class DecorrelatedBatchNormalization(function_node.FunctionNode):

    def __init__(self, groups=16, eps=2e-5, mean=None, projection=None,
                 decay=0.9):
        self.groups = groups

        self.running_mean = mean
        self.running_projection = projection

        self.eps = eps
        self.decay = decay
        self.axis = None

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type = in_types[0]
        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.shape[1] % self.groups == 0,
        )
        type_check.expect(
            x_type.ndim >= 2,
        )

    def forward(self, inputs):
        self.retain_inputs(())
        x = inputs[0]
        xp = backend.get_array_module(x)
        x_shape = x.shape
        b, c = x_shape[:2]
        g = self.groups
        C = c // g
        spatial_axis, m = _calc_axis_and_m(x_shape, b)

        # (g, C, m)
        x_hat = x.transpose((1, 0) + spatial_axis).reshape(g, C, m)
        mean = x_hat.mean(axis=2, keepdims=True)
        x_hat = x_hat - mean
        self.eps = x.dtype.type(self.eps)

        eps_matrix = self.eps * xp.eye(C, dtype=x.dtype)
        cov = _matmul(
            x_hat, x_hat.transpose(0, 2, 1),
            xp) / x.dtype.type(m) + eps_matrix
        # (g, C), (g, C, C)
        self.eigvals, self.eigvectors = _eigh(cov, xp)
        U = _matmul(
            _diag(self.eigvals ** -0.5, xp),
            self.eigvectors.transpose(0, 2, 1),
            xp)
        self.y_hat_pca = _matmul(U, x_hat, xp)  # PCA whitening
        # ZCA whitening
        y_hat = _matmul(self.eigvectors, self.y_hat_pca, xp)

        y = y_hat.reshape((c, b) + x_shape[2:]).transpose(
            (1, 0) + spatial_axis)

        # Update running statistics
        if self.running_mean is not None:
            mean = mean.squeeze(axis=2)
            self.running_mean *= self.decay
            self.running_mean += (1 - self.decay) * mean
        if self.running_projection is not None:
            adjust = m / max(m - 1., 1.)  # unbiased estimation
            self.running_projection *= self.decay
            projection = _matmul(self.eigvectors, U, xp)
            self.running_projection += (1 - self.decay) * adjust * projection

        return y,

    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs

        f = DecorrelatedBatchNormalizationGrad(
            self.groups, self.eigvals, self.eigvectors, self.y_hat_pca)
        return f.apply((gy,))


class DecorrelatedBatchNormalizationGrad(function_node.FunctionNode):

    def __init__(self, groups, eigvals, eigvectors, y_hat_pca):
        self.groups = groups
        self.eigvals = eigvals
        self.eigvectors = eigvectors
        self.y_hat_pca = y_hat_pca

    def forward(self, inputs):
        self.retain_inputs(())
        gy = inputs[0]
        xp = backend.get_array_module(gy)
        gy_shape = gy.shape
        b, c = gy_shape[:2]
        g = self.groups
        C = c // g
        spatial_axis, m = _calc_axis_and_m(gy_shape, b)
        arange_C = numpy.arange(C)
        diag_indices = slice(None), arange_C, arange_C

        gy_hat = gy.transpose((1, 0) + spatial_axis).reshape(g, C, m)

        eigvectors = self.eigvectors
        eigvals = self.eigvals
        y_hat_pca = self.y_hat_pca
        gy_hat_pca = _matmul(eigvectors.transpose(0, 2, 1), gy_hat, xp)
        f = gy_hat_pca.mean(axis=2, keepdims=True)

        K = eigvals[:, :, None] - eigvals[:, None, :]
        valid = K != 0  # to avoid nan, use eig_i != eig_j instead of i != j
        K[valid] = xp.reciprocal(K[valid])

        V = _diag(eigvals, xp)
        V_sqrt = _diag(eigvals ** 0.5, xp)
        V_invsqrt = _diag(eigvals ** -0.5, xp)

        F_c = _matmul(
            gy_hat_pca, y_hat_pca.transpose(0, 2, 1),
            xp) / gy.dtype.type(m)
        M = xp.zeros_like(F_c)
        M[diag_indices] = F_c[diag_indices]

        mat = K.transpose(0, 2, 1) * (
            _matmul(V, F_c.transpose(0, 2, 1), xp)
            + _matmul(_matmul(V_sqrt, F_c, xp), V_sqrt, xp)
        )
        S = mat + mat.transpose(0, 2, 1)
        R = gy_hat_pca - f + _matmul(
            (S - M).transpose(0, 2, 1), y_hat_pca, xp)
        gx_hat = _matmul(
            _matmul(R.transpose(0, 2, 1), V_invsqrt, xp),
            eigvectors.transpose(0, 2, 1), xp
        ).transpose(0, 2, 1)

        gx = gx_hat.reshape((c, b) + gy_shape[2:]).transpose(
            (1, 0) + spatial_axis)

        self.retain_outputs(())
        return gx,

    def backward(self, inputs, grad_outputs):
        # TODO(crcrpar): Implement this.
        raise NotImplementedError('Double backward is not implemented for'
                                  ' decorrelated batch normalization.')


class FixedDecorrelatedBatchNormalization(function_node.FunctionNode):

    def __init__(self, groups):
        self.groups = groups

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        x_type, mean_type, var_type = in_types
        type_check.expect(
            x_type.dtype.kind == 'f',
            mean_type.dtype == x_type.dtype,
            var_type.dtype == x_type.dtype,
        )
        type_check.expect(
            x_type.ndim >= 2,
        )

    def forward(self, inputs):
        self.retain_inputs((0, 1, 2))
        x, mean, projection = inputs
        xp = backend.get_array_module(x)
        x_shape = x.shape
        b, c = x_shape[:2]
        g = self.groups
        C = c // g
        spatial_axis, m = _calc_axis_and_m(x_shape, b)

        x_hat = x.transpose((1, 0) + spatial_axis).reshape(g, C, m)
        x_hat = x_hat - xp.expand_dims(mean, axis=2)

        y_hat = _matmul(projection, x_hat, xp)

        y = y_hat.reshape((c, b) + x_shape[2:]).transpose(
            (1, 0) + spatial_axis)

        return y,

    def backward(self, indexes, grad_outputs):
        x, mean, projection = self.get_retained_inputs()
        gy,  = grad_outputs
        f = FixedDecorrelatedBatchNormalizationGrad(self.groups)
        return f.apply((x, mean, projection, gy))


class FixedDecorrelatedBatchNormalizationGrad(function_node.FunctionNode):

    def __init__(self, groups):
        self.groups = groups

    def forward(self, inputs):
        self.retain_inputs(())
        x, mean, projection, gy = inputs
        xp = backend.get_array_module(x)
        gy_shape = gy.shape
        b, c = gy_shape[:2]
        g = self.groups
        C = c // g
        spatial_axis, m = _calc_axis_and_m(gy_shape, b)

        gy_hat = gy.transpose((1, 0) + spatial_axis).reshape(g, C, m)
        x_hat = x.transpose((1, 0) + spatial_axis).reshape(g, C, m)
        gy_hat_pca = _matmul(projection.transpose(0, 2, 1), gy_hat, xp)
        gx = gy_hat_pca.reshape((c, b) + gy_shape[2:]).transpose(
            (1, 0) + spatial_axis)
        rhs = x_hat - xp.expand_dims(mean, axis=2)
        gprojection = _matmul((x_hat - rhs).transpose(0, 2, 1), gy_hat, xp)
        gmean = -gy_hat_pca[..., 0]
        self.retain_outputs(())
        return gx, gmean, gprojection

    def backward(self, inputs, grad_outputs):
        # TODO(crcrpar): Implement this.
        raise NotImplementedError('Double backward is not implemented for'
                                  ' fixed decorrelated batch normalization.')


def decorrelated_batch_normalization(x, **kwargs):
    """decorrelated_batch_normalization(x, *, groups=16, eps=2e-5, \
running_mean=None, running_projection=None, decay=0.9)

    Decorrelated batch normalization function.

    It takes the input variable ``x`` and normalizes it using
    batch statistics to make the output zero-mean and decorrelated.

    Args:
        x (:class:`~chainer.Variable`): Input variable.
        groups (int): Number of groups to use for group whitening.
        eps (float): Epsilon value for numerical stability.
        running_mean (:ref:`ndarray`): Expected value of the mean. This is a
            running average of the mean over several mini-batches using
            the decay parameter. If ``None``, the expected mean is initialized
            to zero.
        running_projection (:ref:`ndarray`):
            Expected value of the project matrix. This is a
            running average of the projection over several mini-batches using
            the decay parameter. If ``None``, the expected projected is
            initialized to the identity matrix.
        decay (float): Decay rate of moving average. It is used during
            training.

    Returns:
        ~chainer.Variable: The output variable which has the same shape as
        :math:`x`.

    See: `Decorrelated Batch Normalization <https://arxiv.org/abs/1804.08450>`_

    .. seealso:: :class:`~chainer.links.DecorrelatedBatchNormalization`

    """
    groups, eps, running_mean, running_projection, decay = \
        argument.parse_kwargs(
            kwargs, ('groups', 16), ('eps', 2e-5), ('running_mean', None),
            ('running_projection', None), ('decay', 0.9))

    f = DecorrelatedBatchNormalization(
        groups, eps, running_mean, running_projection, decay)
    return f.apply((x,))[0]


def fixed_decorrelated_batch_normalization(x, mean, projection, groups=16):
    """Decorrelated batch normalization function with fixed statistics.

    This is a variant of decorrelated batch normalization, where the mean and
    projection statistics are given by the caller as fixed variables. This is
    used in testing mode of the decorrelated batch normalization layer, where
    batch statistics cannot be used for prediction consistency.

    Args:
        x (:class:`~chainer.Variable`): Input variable.
        mean (:class:`~chainer.Variable` or :ref:`ndarray`):
            Shifting parameter of input.
        projection (:class:`~chainer.Variable` or :ref:`ndarray`):
            Projection matrix for decorrelation of input.
        groups (int): Number of groups to use for group whitening.

    Returns:
        ~chainer.Variable: The output variable which has the same shape as
        :math:`x`.

    .. seealso::
       :func:`~chainer.functions.decorrelated_batch_normalization`,
       :class:`~chainer.links.DecorrelatedBatchNormalization`

    """
    f = FixedDecorrelatedBatchNormalization(groups)
    return f.apply((x, mean, projection))[0]
