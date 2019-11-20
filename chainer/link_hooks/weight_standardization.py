import chainer
from chainer.functions.normalization import group_normalization
from chainer import link_hook


class WeightStandardization(link_hook.LinkHook):
    """Weight Standardization (WS) link hook implementation.

    This hook standardizes a weight by *weight statistics*.

    This link hook implements a WS which computes the mean and variance along
    axis "output channels", then normalizes by these statistics.
    WS improves training by reducing the Lipschitz constants of the loss and
    the gradients like batch normalization (BN) but without relying on large
    batch sizes during training. Specifically, the performance of WS with group
    normalization (GN) trained with small-batch is able to match or outperforms
    that of BN trained with large-batch.
    WS is originally proposed for 2D convolution layers followed by mainly GN
    and sometimes BN.
    Note that this hook is able to handle layers such as N-dimensional
    convolutional, linear and embedding layers but there is no guarantee that
    this hook helps training.

    See: Siyuan Qiao et. al., `Weight Standardization
    <https://arxiv.org/abs/1903.10520>`_

    Args:
        eps (float): Numerical stability in standard deviation calculation.
            The default value is 1e-5.
        weight_name (str): Link's weight name to appky this hook. The default
            value is ``'W'``.
        name (str or None): Name of this hook. The default value is
            ``'WeightStandardization'``.
    """

    name = 'WeightStandardization'

    def __init__(self, *, eps=1e-5, weight_name='W', name=None):
        self.eps = eps
        self.weight_name = weight_name
        self._initialized = False
        if name is not None:
            self.name = name

    def __enter__(self):
        raise NotImplementedError(
            'This hook is not supposed to be used as context manager.')

    def __exit__(self):
        raise NotImplementedError

    def added(self, link):
        if not hasattr(link, self.weight_name):
            raise ValueError(
                'Weight \'{}\' does not exist!'.format(self.weight_name))
        if getattr(link, self.weight_name).array is not None:
            self._initialized = True

    def forward_preprocess(self, cb_args):
        # This method normalizes target link's weight by statistics
        link = cb_args.link
        input_variable = cb_args.args[0]
        if not self._initialized:
            if getattr(link, self.weight_name).array is None:
                if input_variable is None:
                    raise ValueError('Input variable does not exist!')
                link._initialize_params(input_variable.shape[1])
        weight = getattr(link, self.weight_name)
        with chainer.using_device(link.device):
            gamma = link.xp.ones(
                (weight.shape[1],), dtype=weight.dtype)
            beta = link.xp.zeros(
                (weight.shape[1],), dtype=weight.dtype)
        # For link.W or equivalents to be chainer.Parameter
        # consistently to users, this hook maintains a reference to
        # the unnormalized weight.
        self.original_weight = weight
        # note: `normalized_weight` is ~chainer.Variable
        normalized_weight = group_normalization.group_normalization(
            weight, groups=1, gamma=gamma, beta=beta, eps=self.eps)
        setattr(link, self.weight_name, normalized_weight)

    def forward_postprocess(self, cb_args):
        # Here, the computational graph is already created,
        # we can reset link.W or equivalents to be Parameter.
        link = cb_args.link
        setattr(link, self.weight_name, self.original_weight)
