import numpy

from chainer import backend
from chainer.backends import cuda
from chainer import configuration
import chainer.functions as F
import chainer.links as L
from chainer import link_hook
from chainer import variable


def l2normalize(xp, v, eps=1e-12):
    """Normalize a vector by its L2 norm.

    Args:
        xp (numpy or cupy):
        v (numpy.ndarray or cupy.ndarray)
        eps (float): Epsilon value for numerical stability.

    Returns:
        :class:`numpy.ndarray` or :class:`cupy.ndarray`

    """
    if xp is numpy:
        return v / (numpy.linalg.norm(v) + eps)
    else:
        norm = cuda.reduce('T x', 'T out',
                           'x * x', 'a + b', 'out = sqrt(a)', 0,
                           'norm_sn')
        div = cuda.elementwise('T x, T norm, T eps',
                               'T out',
                               'out = x / (norm + eps)',
                               'div_sn')
        return div(v, norm(v), eps)


def update_approximate_vectors(
        weight_matrix, u, n_power_iteration=1, eps=1e-12):
    """Update the first left and right singular vectors.

    This function updates the first left singular vector `u` and
    the first right singular vector `v`.

    Args:
        weight_matrix (variable.Parameter): 2D weight.
        u (numpy.ndarray, cupy.ndarray, or None):
            Vector that has the shape of (1, out_size).
        n_power_iteration (int): Number of iterations to approximate
            the first right and left singular vectors.

    Returns:
        :class:`numpy.ndarray` or `cupy.ndarray`:
            Approximate first left singular vector.
        :class:`numpy.ndarray` or `cupy.ndarray`:
            Approximate first right singular vector.

    """
    weight_matrix = weight_matrix.array
    xp = backend.get_array_module(weight_matrix)
    for _ in range(n_power_iteration):
        v = l2normalize(xp, xp.dot(u, weight_matrix), eps)
        u = l2normalize(xp, xp.dot(v, weight_matrix.T), eps)
    return u, v


def calculate_max_singular_value(weight_matrix, u, v):
    """Calculate max singular value by power iteration method.

    Args:
        weight_matrix (chainer.Variable or chainer.Parameter):
        u (numpy.ndarray or cupy.ndarray)
        v (numpy.ndarray or cupy.ndarray)

    Returns:
        ~chainer.Variable: Max singular value via power iteration method.

    """
    sigma = F.sum(F.linear(u, F.transpose(weight_matrix)) * v)
    return sigma


class SpectralNormalization(link_hook.LinkHook):
    r"""Spectral Normalization link hook implementation.

    This hook normalizes a weight using max singular value and this value
    is computed via power iteration method. Cuurently, this hook is supposed to
    be added to :func:`chainer.links.Linear`, :func:`chainer.links.EmbedID`,
    :func:`chainer.links.Convolution2D`, :func:`chainer.links.ConvolutionND`,
    :func:`chainer.links.Deconvolution2D`,
    and :func:`chainer.links.DeconvolutionND`. However, you can use this to
    other links like RNNs by specifying ``weight_name``.

    .. math::
         \mathbf{W} &:=& \dfrac{\mathbf{W}}{\sigma(\mathbf{W})} \\
         \text{where,} \ \sigma(\mathbf{W}) &:=&
         \max_{\mathbf{h}: \mathbf{h} \ne 0}
         \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}
         = \max_{\|\mathbf{h}\|_2 \le 1} \|\mathbf{W}\mathbf{h}\|_2

    See: T. Miyato et. al., `Spectral Normalization for Generative Adversarial
    Networks <https://arxiv.org/abs/1802.05957>`_

    Args:
        n_power_iteration (int): Number of power iteration.
            The default value is 1.
        eps (int): Numerical stability in norm calculation.
            The default value is 1e-12.
        use_gamma (bool): If ``True``, weight scaling parameter gamma which is
            initialized by initial weight's max singular value is introduced.
        factor (float): Scaling parameter to divide maximum singular value.
            The default value is 1.0.
        weight_name (str): Link's weight name to apply this hook. The default
            value is 'W'.
        name (str or None): Name of this hook. The default value is
            'SpectralNormalization'.

    Attributes:
        vector_name (str): Name of the approximate first left singular vector
            registered in the target link.
            the target link.
            axis (int): Axis of weight represents the numbef of output
                feature maps or output units (``out_channels`` and
                ``out_size``, respectively).

    .. admonition:: Example

        >>> x = np.random.uniform(-1, 1, (10, 3, 32, 32)).astype(np.float32)
        >>> layer = L.Convolution2D(3, 5, 3, 1, 1)
        >>> hook = SpectralNormalization()
        >>> layer.add_hook(hook)
        >>> y = layer(x)

    """

    name = 'SpectralNormalization'

    def __init__(self, n_power_iteration=1, eps=1e-12, use_gamma=False,
                 factor=1.0, weight_name='W', name=None):
        assert n_power_iteration > 0
        self.n_power_iteration = n_power_iteration
        self.eps = eps
        self.use_gamma = use_gamma
        self.factor = factor
        self.weight_name = weight_name
        self.vector_name = weight_name + '_u'
        self._initialied = False
        self.axis = 0

        if name is not None:
            self.name = name

    def added(self, link):
        if isinstance(
            link, (
                L.Deconvolution1D, L.Deconvolution2D,
                L.Deconvolution3D, L.DeconvolutionND)):
            self.axis = 1
        if link.W.array is not None:
            self._prepare_parameters(link)

    def deleted(self, link):
        delattr(link, self.vector_name)
        if self.use_gamma:
            del link.gamma

    def forward_preprocess(self, cb_args):
        if configuration.config.train:
            link = cb_args.link
            input_variable = cb_args.args[0]
            if not self._initialied:
                self._prepare_parameters(link, input_variable)
            weight = getattr(link, self.weight_name)
            normalized_weight = self.normalize_weight(link, weight)
            setattr(link, self.weight_name, normalized_weight)

    def forward_postprocess(self, cb_args):
        if configuration.config.train:
            link = cb_args.link
            weight = getattr(link, self.weight_name)
            delattr(link, self.weight_name)
            link.add_param(self.weight_name, shape=weight.shape,
                           dtype=weight.dtype, initializer=weight.array)

    def _prepare_parameters(self, link, input_variable=None):
        if getattr(link, self.weight_name).array is None:
            if input_variable is not None:
                link._initialize_params(input_variable.shape[1])
        initialW = getattr(link, self.weight_name)
        u = link.xp.random.normal(
            size=(1, initialW.shape[self.axis])).astype(dtype=link.W.dtype)
        setattr(link, self.vector_name, u)
        link.register_persistent(self.vector_name)
        if self.use_gamma:
            # Initialize the scaling parameter with the max singular value.
            weight_matrix = self._reshape_W(initialW.array)
            _, s, _ = link.xp.linalg.svd(weight_matrix)
            gamma_shape = [1] * initialW.ndim
            with link.init_scope():
                link.gamma = variable.Parameter(s[0], gamma_shape)
        self._initialied = True

    def normalize_weight(self, link, *args, **kwargs):
        weight_name, vector_name = self.weight_name, self.vector_name
        W = getattr(link, weight_name)
        u = getattr(link, vector_name)
        weight_matrix = self._reshape_W(W)
        u, v = update_approximate_vectors(
            weight_matrix, u, self.n_power_iteration, self.eps)
        sigma = calculate_max_singular_value(weight_matrix, u, v) / self.factor
        if self.use_gamma:
            W = F.broadcast_to(link.gamma, W.shape) * W / sigma
        else:
            W = W / sigma
        link.xp.copyto(u, getattr(link, vector_name))
        return W

    def _reshape_W(self, W):
        """Reshape & transpose weight into 2D if necessary."""
        if W.ndim == 2:
            return W
        if self.axis != 0:
            axes = [self.axis] + [i for i in range(W.ndim) if i != self.axis]
            W = W.transpose(axes)
        return W.reshape(W.shape[0], -1)
