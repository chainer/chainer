from chainer import backend
from chainer import configuration
import chainer.functions as F
from chainer import link_hook
import chainer.links as L
from chainer import variable

class WeightStandardization(link_hook.LinkHook):
    """Weight Standardization link hook implementation.

    This hook standardize a weight by *weight statistics*.

    See: Siyuan Qiao et. al., `Weight Standardization
    <https://arxiv.org/abs/1903.10520>`_

    Args:
    eps (int): Numerical stability in standard deviation calculation.
        The default value is 1e-5.
    weight_name (str): Link's weight name to appky this hook. The default
        value is ``'W'``.
    name (str or None): Name of this hook. The default value is
        ``'WeightStandardization'``.
    """

    name = 'WeightStandardization'

    def __init__(self, eps=1e-5, weight_name='W', name=None):
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
        # For link.W or equivalents to be chainer.Parameter
        # consistently to users, this hook maintains a reference to
        # the unnormalized weight.
        self.original_weight = weight
        # note: `normalized_weight` is ~chainer.Variable
        normalized_weight = F.standardize(weight, self.eps)
        setattr(link, self.weight_name, normalized_weight)

    def forward_postprocess(self, cb_args):
        # Here, the computational graph is already created,
        # we can reset link.W or equivalents to be Parameter.
        link = cb_args.link
        setattr(link, self.weight_name, self.original_weight)
