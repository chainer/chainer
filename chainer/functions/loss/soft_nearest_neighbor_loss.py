import numpy

from chainer.backends import cuda
from chainer import functions as F

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

    batch = x.data.shape[0]
    dtype = numpy.float32
    tmp_matrix = numpy.ones((batch, batch), dtype=dtype) - \
        numpy.eye(batch, dtype=dtype)
    xp = x.xp

    if xp != numpy:
        tmp_matrix = cuda.to_gpu(tmp_matrix)
    f = fits(x, x, temp, cos_distance) * tmp_matrix
    return f / (
        STABILITY_EPS + F.expand_dims(F.sum(f, 1), 1))


def same_label_mask(y, y2, xp):
    """Masking matrix such that element i,j is 1 iff y[i] == y2[i].

    :param y: a list of labels
    :param y2: a list of labels

    :returns: A tensor for the masking matrix.
    """
    return xp.squeeze(xp.equal(y, xp.expand_dims(y2, 1)))
#    return xp.squeeze(xp.equal(y.data, xp.expand_dims(y2.data, 1)))


def masked_pick_probability(x, y, temp, cos_distance):
    """The pairwise sampling probabilities for the elements of x for neighbor

    points which share labels.
    :param x: a matrix
    :param y: a list of labels for each element of x
    :param temp: Temperature
    :cos_distance: Boolean for using cosine or Euclidean distance

    :returns: A tensor for the pairwise sampling probabilities.
    """

    return pick_probability(x, temp, cos_distance) * \
        same_label_mask(y, y, x.xp)


def soft_nearest_neighbor_loss(x, y, temp, cos_distance):
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
