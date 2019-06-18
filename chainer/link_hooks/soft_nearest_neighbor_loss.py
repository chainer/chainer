import chainer
from chainer import variable
from chainer import link_hook
import chainer.functions as F


def optimized_temp_SNNL(x, y, initial_temp, cos_distance):
    """The optimized variant of Soft Nearest Neighbor Loss.

    Every time this tensor is evaluated, the temperature is optimized
    to minimize the loss value, this results in more numerically stable
    calculations of the SNNL.
    :param x: a matrix.
    :param y: a list of labels for each element of x.
    :param initial_temp: Temperature.
    :cos_distance: Boolean for using cosine or Euclidean distance.

    :returns: A tensor for the Soft Nearest Neighbor Loss of the points
              in x with labels y, optimized for temperature.
    """
    xp = x.xp
    t = variable.Variable(xp.asarray([1], dtype=xp.float32))

    def inverse_temp(t):
        # pylint: disable=missing-docstring
        # we use inverse_temp because it was observed to be more stable
        # when optimizing.
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
