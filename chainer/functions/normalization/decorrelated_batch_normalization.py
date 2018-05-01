from chainer.backends import cuda
from chainer import function
from chainer import function_node
from chainer.utils import argument
from chainer.utils import type_check


class DecorrelatedBatchNormalizationFunction(function_node.FunctionNode):

    def __init__(self, groups=16, eps=2e-5, mean=None, projection=None,
                 decay=0.9):
        self.groups = groups

        self.expected_mean = mean
        self.expected_projection = projection

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
        xp = cuda.get_array_module(x)
        spatial_ndim = len(x.shape[2:])
        spatial_axis = tuple(range(2, 2 + spatial_ndim))
        b, c = x.shape[:2]
        g = self.groups
        C = c // g
        m = b * g
        for i in spatial_axis:
            m *= x.shape[i]

        if self.expected_mean is None:
            self.expected_mean = xp.zeros(C, dtype=x.dtype)
            self.expected_projection = xp.eye(C, dtype=x.dtype)

        if self.groups > 1:
            x = x.reshape((b * g, C, ) + x.shape[2:])
        x_hat = x.transpose((1, 0) + spatial_axis).reshape((C, -1))

        mean = x_hat.mean(axis=1)
        x_hat = x_hat - mean[:, None]
        self.eps = x.dtype.type(self.eps)

        I = self.eps * xp.eye(C, dtype=x.dtype)
        cov = x_hat.dot(x_hat.T) / x.dtype.type(m) + I
        self.eigvals, self.eigvectors = xp.linalg.eigh(cov)
        U = xp.diag(self.eigvals ** -0.5).dot(self.eigvectors.T)
        self.y_hat_pca = U.dot(x_hat)  # PCA whitening
        y_hat = self.eigvectors.dot(self.y_hat_pca)  # ZCA whitening

        y = y_hat.reshape((C, b * g,) + x.shape[2:]).transpose(
            (1, 0) + spatial_axis)
        if self.groups > 1:
            y = y.reshape((-1, c, ) + x.shape[2:])

        # Update running statistics
        q = x.size // C
        adjust = q / max(q - 1., 1.)  # unbiased estimation
        self.expected_mean *= self.decay
        self.expected_mean += (1 - self.decay) * mean
        self.expected_projection *= self.decay
        projection = self.eigvectors.dot(U)
        self.expected_projection += (1 - self.decay) * adjust * projection

        return y,

    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs

        f = DecorrelatedBatchNormalizationGrad(
            self.groups, self.eigvals, self.eigvectors, self.y_hat_pca)
        return f(gy),


class DecorrelatedBatchNormalizationGrad(function.Function):

    def __init__(self, groups, eigvals, eigvectors, y_hat_pca):
        self.groups = groups
        self.eigvals = eigvals
        self.eigvectors = eigvectors
        self.y_hat_pca = y_hat_pca

    def forward(self, inputs):
        self.retain_inputs(())
        gy = inputs[0]
        xp = cuda.get_array_module(gy)
        spatial_ndim = len(gy.shape[2:])
        spatial_axis = tuple(range(2, 2 + spatial_ndim))
        b, c = gy.shape[:2]
        g = self.groups
        C = c // g
        m = b * g
        for i in spatial_axis:
            m *= gy.shape[i]

        if self.groups > 1:
            gy = gy.reshape((b * g, C, ) + gy.shape[2:])
        gy_hat = gy.transpose((1, 0) + spatial_axis).reshape((C, -1))

        gy_hat_pca = self.eigvectors.T.dot(gy_hat)
        f = gy_hat_pca.mean(axis=1)

        K = self.eigvals[:, None] - self.eigvals[None, :]
        valid = K != 0
        K[valid] = 1 / K[valid]
        xp.fill_diagonal(K, 0)

        V = xp.diag(self.eigvals)
        V_sqrt = xp.diag(self.eigvals ** 0.5)
        V_invsqrt = xp.diag(self.eigvals ** -0.5)

        F_c = gy_hat_pca.dot(self.y_hat_pca.T) / gy.dtype.type(m)
        M = xp.diag(xp.diag(F_c))

        S = 2 * _sym(K.T * (V.dot(F_c.T) + V_sqrt.dot(F_c).dot(V_sqrt)))
        R = gy_hat_pca - f[:, None] + (S - M).T.dot(self.y_hat_pca)
        gx_hat = R.T.dot(V_invsqrt).dot(self.eigvectors.T).T

        gx = gx_hat.reshape((C, b * g,) + gy.shape[2:]).transpose(
            (1, 0) + spatial_axis)
        if self.groups > 1:
            gx = gx.reshape((-1, c, ) + gy.shape[2:])

        self.retain_outputs(())
        return gx,

    def backward(self, inputs, grad_outputs):
        raise NotImplementedError


class FixedDecorrelatedBatchNormalizationFunction(function_node.FunctionNode):

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
        x, mean, projection = inputs
        spatial_ndim = len(x.shape[2:])
        spatial_axis = tuple(range(2, 2 + spatial_ndim))
        b, c = x.shape[:2]
        g = self.groups
        C = c // g

        if self.groups > 1:
            x = x.reshape((b * g, C, ) + x.shape[2:])
        x_hat = x.transpose((1, 0) + spatial_axis).reshape((C, -1))

        y_hat = projection.dot(x_hat - mean[:, None])

        y = y_hat.reshape((C, b * g,) + x.shape[2:]).transpose(
            (1, 0) + spatial_axis)
        if self.groups > 1:
            y = y.reshape((-1, c, ) + x.shape[2:])

        return y,

    def backward(self, indexes, grad_outputs):
        raise NotImplementedError


def _sym(x):
    return (x + x.T) / 2


def decorrelated_batch_normalization(x, **kwargs):
    groups, eps, expected_mean, expected_projection, decay = \
        argument.parse_kwargs(
            kwargs, ('groups', 16), ('eps', 2e-5), ('expected_mean', None),
            ('expected_projection', None), ('decay', 0.9))

    f = DecorrelatedBatchNormalizationFunction(groups, eps, expected_mean,
                                               expected_projection,
                                               decay)
    return f.apply((x,))[0]


def fixed_decorrelated_batch_normalization(x, mean, projection, **kwargs):
    groups, = argument.parse_kwargs(
        kwargs, ('groups', 16))

    f = FixedDecorrelatedBatchNormalizationFunction(groups)
    return f.apply((x, mean, projection))[0]
