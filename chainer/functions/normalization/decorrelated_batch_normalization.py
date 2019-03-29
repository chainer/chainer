from chainer import backend
from chainer import function_node
from chainer.utils import argument
from chainer.utils import type_check


def _calc_axis_and_m(x_shape, batch_size, groups):
    m = batch_size * groups
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
        spatial_axis, m = _calc_axis_and_m(x_shape, b, g)

        if g > 1:
            x = x.reshape((b * g, C) + x.shape[2:])
        x_hat = x.transpose((1, 0) + spatial_axis).reshape(C, -1)

        mean = x_hat.mean(axis=1)
        x_hat = x_hat - mean[:, None]
        self.eps = x.dtype.type(self.eps)

        eps_matrix = self.eps * xp.eye(C, dtype=x.dtype)
        cov = x_hat.dot(x_hat.T) / x.dtype.type(m) + eps_matrix
        self.eigvals, self.eigvectors = xp.linalg.eigh(cov)
        U = xp.diag(self.eigvals ** -0.5).dot(self.eigvectors.T)
        self.y_hat_pca = U.dot(x_hat)  # PCA whitening
        y_hat = self.eigvectors.dot(self.y_hat_pca)  # ZCA whitening

        y = y_hat.reshape((C, b * g,) + x.shape[2:]).transpose(
            (1, 0) + spatial_axis)
        if self.groups > 1:
            y = y.reshape((-1, c) + x.shape[2:])

        # Update running statistics
        if self.running_mean is not None:
            self.running_mean *= self.decay
            self.running_mean += (1 - self.decay) * mean
        if self.running_projection is not None:
            adjust = m / max(m - 1., 1.)  # unbiased estimation
            self.running_projection *= self.decay
            projection = self.eigvectors.dot(U)
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
        spatial_axis, m = _calc_axis_and_m(gy_shape, b, g)

        if g > 1:
            gy = gy.reshape((b * g, C) + gy.shape[2:])
        gy_hat = gy.transpose((1, 0) + spatial_axis).reshape(C, -1)

        eigvectors = self.eigvectors
        eigvals = self.eigvals
        y_hat_pca = self.y_hat_pca
        gy_hat_pca = eigvectors.T.dot(gy_hat)
        f = gy_hat_pca.mean(axis=1)

        K = eigvals[:, None] - eigvals[None, :]
        valid = K != 0
        K[valid] = 1 / K[valid]
        xp.fill_diagonal(K, 0)

        V = xp.diag(eigvals)
        V_sqrt = xp.diag(eigvals ** 0.5)
        V_invsqrt = xp.diag(eigvals ** -0.5)

        F_c = gy_hat_pca.dot(y_hat_pca.T) / gy.dtype.type(m)
        M = xp.diag(xp.diag(F_c))

        mat = K.T * (V.dot(F_c.T) + V_sqrt.dot(F_c).dot(V_sqrt))
        S = mat + mat.T
        R = gy_hat_pca - f[:, None] + (S - M).T.dot(y_hat_pca)
        gx_hat = R.T.dot(V_invsqrt).dot(eigvectors.T).T

        gx = gx_hat.reshape((C, b * g,) + gy.shape[2:]).transpose(
            (1, 0) + spatial_axis)
        if g > 1:
            gx = gx.reshape((-1, c, ) + gy.shape[2:])

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
        x_shape = x.shape
        b, c = x_shape[:2]
        g = self.groups
        C = c // g
        spatial_axis, m = _calc_axis_and_m(x_shape, b, g)

        if g > 1:
            x = x.reshape((b * g, C) + x.shape[2:])
        x_hat = x.transpose((1, 0) + spatial_axis).reshape(C, -1)

        y_hat = projection.dot(x_hat - mean[:, None])

        y = y_hat.reshape((C, b * g) + x.shape[2:]).transpose(
            (1, 0) + spatial_axis)
        if g > 1:
            y = y.reshape((-1, c) + x.shape[2:])

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
        gy_shape = gy.shape
        b, c = gy_shape[:2]
        g = self.groups
        C = c // g
        spatial_axis, m = _calc_axis_and_m(gy_shape, b, g)

        if g > 1:
            gy = gy.reshape((b * g, C) + gy.shape[2:])
            x = x.reshape((b * g, C) + x.shape[2:])
        x_hat = x.transpose((1, 0) + spatial_axis).reshape(C, -1)
        gy_hat = gy.transpose((1, 0) + spatial_axis).reshape(C, -1)
        gy_hat_pca = projection.T.dot(gy_hat)
        gx = gy_hat_pca.reshape(
            (C, b * g) + gy.shape[2:]).transpose((1, 0) + spatial_axis)
        if g > 1:
            gx = gx.reshape((-1, c) + gy.shape[2:])
        rhs = x_hat - mean[Ellipsis, None]
        gprojection = (x_hat - rhs).T.dot(gy_hat)
        gmean = -gx[:, 0]
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
