import numpy

from chainer.backends import cuda
import chainer.functions

STABILITY_EPS = 0.00001


def pairwise_euclid_distance(A, B):
    # Pairwise Euclidean distance between two matrices.
    F = chainer.functions
    batchA = A.data.shape[0]
    batchB = B.data.shape[0]

    sqr_norm_A = F.reshape(F.sum(F.square(A), axis=1), (1, batchA))
    sqr_norm_B = F.reshape(F.sum(F.square(B), axis=1), (batchB, 1))
    inner_prod = F.matmul(B, A, transb=True)

    tile_1 = F.tile(sqr_norm_A, (batchB, 1))
    tile_2 = F.tile(sqr_norm_B, (1, batchA))
    return (tile_1 + tile_2 - 2 * inner_prod)


def pairwise_cos_distance(A, B):
    # Pairwise cosine distance between two matrices.
    F = chainer.functions
    normalized_A = F.normalize(A, axis=1)
    normalized_B = F.normalize(B, axis=1)
    prod = F.matmul(normalized_A, normalized_B, transb=True)
    return 1 - prod


def fits(A, B, temp, cos_distance):
    # Exponentiated pairwise distance between each element of A and B.
    if cos_distance:
        distance_matrix = pairwise_cos_distance(A, B)
    else:
        distance_matrix = pairwise_euclid_distance(A, B)
    return chainer.functions.exp(-(distance_matrix / temp))


def pick_probability(x, temp, cos_distance):
    # Row normalized exponentiated pairwise distance between all the elements
    F = chainer.functions
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
    # Masking matrix such that element i,j is 1 iff y[i] == y2[i].
    return xp.squeeze(xp.equal(y, xp.expand_dims(y2, 1)))


def masked_pick_probability(x, y, temp, cos_distance):
    # The pairwise sampling probabilities for the elements of x for neighbor
    return pick_probability(x, temp, cos_distance) * \
        same_label_mask(y, y, x.xp)


def soft_nearest_neighbor_loss(x, y, temp, cos_distance):
    """Computes soft nearest neighbor loss.

    See: `Analyzing and Improving Representations
    with the Soft Nearest Neighbor Loss
    <https://arxiv.org/abs/1902.01889>`_.


    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Variable holding a multidimensional array whose element indicates
            unnormalized log probability: the first axis of the variable
            represents the number of samples, and the second axis represents
            the number of classes.
        t (:class:`~chainer.Variable` or :ref:`ndarray`):
            Variable holding a signed integer vector of ground truth
            labels.
        temp (float32): a temperature
        cos_distance (bool): Boolean for using cosine or Euclidean distance.

    Returns:
        ~chainer.Variable: A variable holding a scalar array of the soft
        nearest neighbor loss of the points in x with labels y.

    """
    F = chainer.functions
    summed_masked_pick_prob = F.sum(
        masked_pick_probability(x, y, temp, cos_distance), 1)
    return F.mean(
        -F.log(STABILITY_EPS + summed_masked_pick_prob))
