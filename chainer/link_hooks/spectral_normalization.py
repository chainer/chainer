import chainer
from chainer import backend
from chainer import configuration
import chainer.functions as F
from chainer import link_hook
import chainer.links as L
from chainer import variable
import chainerx


def l2normalize(xp, v, eps):
    """Normalize a vector by its L2 norm.

    Args:
        xp (numpy or cupy):
        v (numpy.ndarray or cupy.ndarray)
        eps (float): Epsilon value for numerical stability.

    Returns:
        :class:`numpy.ndarray` or :class:`cupy.ndarray`

    """
    # TODO(crcrpar): Remove this when chainerx.linalg.norm becomes available.
    if xp is chainerx:
        # NOTE(crcrpar): `chainerx.power` is not available as of 2019/03/27.
        # See https://github.com/chainer/chainer/pull/6522
        norm = chainerx.sqrt(chainerx.sum(v * v))
    else:
        norm = xp.linalg.norm(v)
    return v / (norm + eps)


def update_approximate_vectors(
        weight_matrix, u, n_power_iteration, eps):
    """Update the first left and right singular vectors.

    This function updates the first left singular vector `u` and
    the first right singular vector `v`.

    Args:
        weight_matrix (~chainer.Variable): 2D weight.
        u (numpy.ndarray, cupy.ndarray, or None):
            Vector that approximates the first left singular vector and
            has the shape of (out_size,).
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
        u = l2normalize(xp, xp.dot(weight_matrix, v), eps)
    return u, v


def calculate_max_singular_value(weight_matrix, u, v):
    """Calculate max singular value by power iteration method.

    Args:
        weight_matrix (~chainer.Variable)
        u (numpy.ndarray or cupy.ndarray)
        v (numpy.ndarray or cupy.ndarray)

    Returns:
        ~chainer.Variable: Max singular value via power iteration method.

    """
    sigma = F.matmul(F.matmul(u, weight_matrix), v)
    return sigma


class SpectralNormalization(link_hook.LinkHook):
    """Spectral Normalization link hook implementation.

    This hook normalizes a weight using max singular value and this value
    is computed via power iteration method. Currently, this hook is supposed to
    be added to :class:`chainer.links.Linear`, :class:`chainer.links.EmbedID`,
    :class:`chainer.links.Convolution2D`, :class:`chainer.links.ConvolutionND`,
    :class:`chainer.links.Deconvolution2D`,
    and :class:`chainer.links.DeconvolutionND`. However, you can use this to
    other links like RNNs by specifying ``weight_name``.
    It is highly recommended to add this hook before optimizer setup because
    this hook add a scaling parameter ``gamma`` if ``use_gamma`` is True.
    Otherwise, the registered ``gamma`` will not be updated.

    .. math::

       \\bar{\\mathbf{W}} &=& \\dfrac{\\mathbf{W}}{\\sigma(\\mathbf{W})} \\\\
       \\text{, where} \\ \\sigma(\\mathbf{W}) &:=&
        \\max_{\\mathbf{h}: \\mathbf{h} \\ne 0}
       \\dfrac{\\|\\mathbf{W} \\mathbf{h}\\|_2}{\\|\\mathbf{h}\\|_2}
        = \\max_{\\|\\mathbf{h}\\|_2 \\le 1} \\|\\mathbf{W}\\mathbf{h}\\|_2

    See: T. Miyato et. al., `Spectral Normalization for Generative Adversarial
    Networks <https://arxiv.org/abs/1802.05957>`_

    Args:
        n_power_iteration (int): Number of power iteration.
            The default value is 1.
        eps (float): Numerical stability in norm calculation.
            The default value is 1e-6 for the compatibility with
            mixed precision training. The value used in the author's
            implementation is 1e-12.
        use_gamma (bool): If ``True``, weight scaling parameter gamma which is
            initialized by initial weight's max singular value is introduced.
        factor (float, None): Scaling parameter to divide maximum singular
            value.  The default value is 1.0.
        weight_name (str): Link's weight name to apply this hook. The default
            value is ``'W'``.
        name (str or None): Name of this hook. The default value is
            ``'SpectralNormalization'``.

    Attributes:
        vector_name (str): Name of the approximate first left singular vector
            registered in the target link.
            the target link.
        axis (int): Axis of weight represents the number of output
            feature maps or output units (``out_channels`` and
            ``out_size``, respectively).

    .. admonition:: Example

        There are almost the same but 2 ways to apply spectral normalization
        (SN) hook to links.

        1. Initialize link and SN separately. This makes it easy to handle
        buffer and parameter of links registered by SN hook.

            >>> l = L.Convolution2D(3, 5, 3)
            >>> hook = chainer.link_hooks.SpectralNormalization()
            >>> _ = l.add_hook(hook)
            >>> # Check the shape of the first left singular vector.
            >>> getattr(l, hook.vector_name).shape
            (5,)
            >>> # Delete SN hook from this link.
            >>> l.delete_hook(hook.name)

        2. Initialize both link and SN hook at one time. This makes it easy to
        define your original :class:`~chainer.Chain`.

            >>> # SN hook handles lazy initialization!
            >>> layer = L.Convolution2D(
            ...     5, 3, stride=1, pad=1).add_hook(
            ...         chainer.link_hooks.SpectralNormalization())
    """

    name = 'SpectralNormalization'

    def __init__(self, n_power_iteration=1, eps=1e-6, use_gamma=False,
                 factor=None, weight_name='W', name=None):
        assert n_power_iteration > 0
        self.n_power_iteration = n_power_iteration
        self.eps = eps
        self.use_gamma = use_gamma
        self.factor = factor
        self.weight_name = weight_name
        self.vector_name = weight_name + '_u'
        self._initialized = False
        self.axis = 0
        if name is not None:
            self.name = name

    def __enter__(self):
        raise NotImplementedError(
            'This hook is not supposed to be used as context manager.')

    def __exit__(self):
        raise NotImplementedError

    def added(self, link):
        # Define axis and register ``u`` if the weight is initialized.
        if not hasattr(link, self.weight_name):
            raise ValueError(
                'Weight \'{}\' does not exist!'.format(self.weight_name))
        if isinstance(link, (L.Deconvolution2D, L.DeconvolutionND)):
            self.axis = 1
        if getattr(link, self.weight_name).array is not None:
            self._prepare_parameters(link)

    def deleted(self, link):
        # Remove approximate vector ``u`` and parameter ``gamma` if exists.
        delattr(link, self.vector_name)
        if self.use_gamma:
            del link.gamma

    def forward_preprocess(self, cb_args):
        # This method normalizes target link's weight spectrally
        # using power iteration method
        link = cb_args.link
        input_variable = cb_args.args[0]
        if not self._initialized:
            self._prepare_parameters(link, input_variable)
        weight = getattr(link, self.weight_name)
        # For link.W or equivalents to be chainer.Parameter
        # consistently to users, this hook maintains a reference to
        # the unnormalized weight.
        self.original_weight = weight
        # note: `normalized_weight` is ~chainer.Variable
        normalized_weight = self.normalize_weight(link)
        setattr(link, self.weight_name, normalized_weight)

    def forward_postprocess(self, cb_args):
        # Here, the computational graph is already created,
        # we can reset link.W or equivalents to be Parameter.
        link = cb_args.link
        setattr(link, self.weight_name, self.original_weight)

    def _prepare_parameters(self, link, input_variable=None):
        """Prepare one buffer and one parameter.

        Args:
            link (:class:`~chainer.Link`): Link to normalize spectrally.
            input_variable (:class:`~chainer.Variable`):
                The first minibatch to initialize weight.

        """
        if getattr(link, self.weight_name).array is None:
            if input_variable is not None:
                link._initialize_params(input_variable.shape[1])
        initialW = getattr(link, self.weight_name)
        if initialW.shape[self.axis] == 0:
            raise ValueError(
                'Expect {}.shape[{}] > 0'.format(self.weight_name, self.axis)
            )
        u = link.xp.random.normal(
            size=(initialW.shape[self.axis],)).astype(dtype=initialW.dtype)
        setattr(link, self.vector_name, u)
        link.register_persistent(self.vector_name)
        if self.use_gamma:
            # Initialize the scaling parameter with the max singular value.
            weight_matrix = self.reshape_W(initialW.array)
            # TODO(crcrpar): Remove this when chainerx supports SVD.
            device = link.device
            if device.xp is chainerx:
                fallback_device = device.fallback_device
                weight_matrix_ = fallback_device.send(weight_matrix)
                with chainer.using_device(fallback_device):
                    _, s_, _ = fallback_device.xp.linalg.svd(weight_matrix_)
                s = device.send(s_)
            else:
                _, s, _ = link.xp.linalg.svd(weight_matrix)
            s0 = chainer.utils.force_array(s[0])
            with link.init_scope():
                link.gamma = variable.Parameter(s0)
        self._initialized = True

    def normalize_weight(self, link):
        """Normalize target weight before every single forward computation."""
        weight_name, vector_name = self.weight_name, self.vector_name
        W = getattr(link, weight_name)
        u = getattr(link, vector_name)
        weight_matrix = self.reshape_W(W)
        if not configuration.config.in_recomputing:
            with chainer.using_device(link.device):
                u, v = update_approximate_vectors(
                    weight_matrix, u, self.n_power_iteration, self.eps)
        else:
            v = self.v
        sigma = calculate_max_singular_value(weight_matrix, u, v)
        if self.factor is not None:
            sigma /= self.factor
        if self.use_gamma:
            W = link.gamma * W / sigma
        else:
            W = W / sigma
        if not configuration.config.in_recomputing:
            self.v = v
            with chainer.using_device(link.device):
                if configuration.config.train:
                    if link.xp is chainerx:
                        # TODO(crcrpar): Remove this when
                        # chainerx supports `copyto`.
                        getattr(link, vector_name)[:] = u
                    else:
                        backend.copyto(getattr(link, vector_name), u)
        return W

    def reshape_W(self, W):
        """Reshape & transpose weight into 2D if necessary."""
        if self.axis != 0:
            axes = [self.axis] + [i for i in range(W.ndim) if i != self.axis]
            W = W.transpose(axes)
        if W.ndim == 2:
            return W
        return W.reshape(W.shape[0], -1)
