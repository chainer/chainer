import chainer
import chainer.functions as F
from chainer import link_hook
from chainer import variable


def optimized_temp_SNNL(x, y, initial_temp, cos_distance):
    """The optimized variant of soft nearest neighbor loss.

    Every time this function is evaluated, the temperature is optimized
    to minimize the loss value, this results in more numerically stable
    calculations of the SNNL.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            A representation of the raw input viector in some hidden layer.
        t (:class:`~chainer.Variable` or :ref:`ndarray`):
            Variable holding a signed integer vector of ground truth
            labels.
        initial_temp (float32): a temperature
        cos_distance (bool): Boolean for using cosine or Euclidean distance.

    Returns:
        ~chainer.Variable: A variable holding a scalar array of the soft
        nearest neighbor loss at a optimized temperature.

    """
    xp = x.xp
    t = variable.Variable(xp.asarray([1], dtype=xp.float32))

    def inverse_temp(t):
        return initial_temp / t
    ent_loss = F.soft_nearest_neighbor_loss(
        x, y, inverse_temp(t), cos_distance)

    grad_t = chainer.grad([ent_loss], [t])[0]
    if grad_t is not None:
        updated_t = t - 0.1 * grad_t
    else:
        updated_t = t

    inverse_t = inverse_temp(updated_t).data

    return F.soft_nearest_neighbor_loss(x, y, inverse_t, cos_distance)


class SNNL_hook(link_hook.LinkHook):
    """link hook for calculate the soft nearest neighbor loss at hidden layer.

    Args:
        temperature (float32): Temperature used for SNNL.
        optimize_temperature (bool):
            Optimize temperature at each calculation to minimize the loss.
            This makes the loss more stable.
        cos_distance (bool): Use cosine distance when calculating SNNL.
    """

    name = 'SNNL_hook'

    def __init__(self, temperature=100.,
                 optimize_temperature=True, cos_distance=False):
        self.temperature = temperature
        self.optimize_temperature = optimize_temperature
        self.cos_distance = cos_distance

    def set_t(self, t):
        self.t = t

    def forward_postprocess(self, args):
        out = args.out
        if self.optimize_temperature is True:
            self.loss = optimized_temp_SNNL(
                out, self.t, self.temperature, self.cos_distance)
        else:
            self.loss = F.soft_nearest_neighbor_loss(
                out, self.t, self.temperature, self.cos_distance)

    def get_loss(self):
        return self.loss
