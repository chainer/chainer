import numpy

import chainer
import chainer.functions as F
from chainer import link_hook
from chainer import Variable

import numpy as np

STABILITY_EPS = 0.00001


def pairwise_euclid_distance(A, B):
    """Pairwise Euclidean distance between two matrices.

    :param A: a matrix.
    :param B: a matrix.
    :returns: A tensor for the pairwise Euclidean between A and B.

    """

    batchA = A.data.shape[0]
    batchB = B.data.shape[0]

    sqr_norm_A = F.reshape(F.sum(F.square(A), axis=1), (1, batchA))
    sqr_norm_B = F.reshape(F.sum(F.square(B), axis=1), (batchB, 1))
    inner_prod = F.matmul(B, A, transb=True)

    tile_1 = F.tile(sqr_norm_A, (batchB, 1))
    tile_2 = F.tile(sqr_norm_B, (1, batchA))
    return (tile_1 + tile_2 - 2 * inner_prod)


def pairwise_cos_distance(A, B):
    """Pairwise cosine distance between two matrices.

    :param A: a matrix.
    :param B: a matrix.

    :returns: A tensor for the pairwise cosine between A and B.

    """

    normalized_A = F.normalize(A, axis=1)
    normalized_B = F.normalize(B, axis=1)
    prod = F.matmul(normalized_A, normalized_B, transb=True)
    return 1 - prod


def fits(A, B, temp, cos_distance):
    """Exponentiated pairwise distance between each element of A and B.

    :param A: a matrix.
    :param B: a matrix.
    :param temp: Temperature
    :cos_distance: Boolean for using cosine or Euclidean distance.

    :returns: A tensor for the exponentiated pairwise distance between
    each element and A and all those of B.

    """

    if cos_distance:
        distance_matrix = pairwise_cos_distance(A, B)
    else:
        distance_matrix = pairwise_euclid_distance(A, B)
    return F.exp(-(distance_matrix / temp))


def pick_probability(x, temp, cos_distance):
    """Row normalized exponentiated pairwise distance between all the elements

    of x. Conceptualized as the probability of sampling a neighbor point for
    every element of x, proportional to the distance between the points.
    :param x: a matrix
    :param temp: Temperature
    :cos_distance: Boolean for using cosine or euclidean distance

    :returns: A tensor for the row normalized exponentiated pairwise distance
              between all the elements of x.

    """

    xp = x.xp
    f = fits(x, x, temp, cos_distance) * (
        xp.ones((x.data.shape[0], x.data.shape[0]), dtype=xp.float32) -
        xp.eye(x.data.shape[0], dtype=xp.float32))
    return f / (
        STABILITY_EPS + F.expand_dims(F.sum(f, 1), 1))


def same_label_mask(y, y2):
    """Masking matrix such that element i,j is 1 iff y[i] == y2[i].

    :param y: a list of labels
    :param y2: a list of labels

    :returns: A tensor for the masking matrix.
    """
    xp = np
    return xp.squeeze(xp.equal(y.data, xp.expand_dims(y2.data, 1)))


def masked_pick_probability(x, y, temp, cos_distance):
    """The pairwise sampling probabilities for the elements of x for neighbor

    points which share labels.
    :param x: a matrix
    :param y: a list of labels for each element of x
    :param temp: Temperature
    :cos_distance: Boolean for using cosine or Euclidean distance

    :returns: A tensor for the pairwise sampling probabilities.
    """
    return pick_probability(x, temp, cos_distance) * same_label_mask(y, y)


def SNNL(x, y, temp, cos_distance):
    """Soft Nearest Neighbor Loss

    :param x: a matrix.
    :param y: a list of labels for each element of x.
    :param temp: Temperature.
    :cos_distance: Boolean for using cosine or Euclidean distance.

    :returns: A tensor for the Soft Nearest Neighbor Loss of the points
              in x with labels y.
    """
    summed_masked_pick_prob = F.sum(
        masked_pick_probability(x, y, temp, cos_distance), 1)
#    tmp = np.log(STABILITY_EPS + summed_masked_pick_prob.data)
#    if np.isnan(tmp).any():
#        print(summed_masked_pick_prob)
#        exit(1)
    return F.mean(
        -F.log(STABILITY_EPS + summed_masked_pick_prob))


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
    t_np = np.asarray([1], dtype=np.float32)
    t = Variable(t_np)

    def inverse_temp(t):
        # pylint: disable=missing-docstring
        # we use inverse_temp because it was observed to be more stable
        # when optimizing.
        return initial_temp / t
    ent_loss = SNNL(x, y, inverse_temp(t), cos_distance)

    grad_t = chainer.grad([ent_loss], [t])[0]
    if grad_t is not None:
        updated_t = t - 0.1 * grad_t
    else:
        updated_t = t

    inverse_t = inverse_temp(updated_t).data

    return SNNL(x, y, inverse_t, cos_distance)


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
            self.loss = SNNL(
                out, self.t, self.temperature, self.cos_distance)

    def get_loss(self):
        return self.loss
