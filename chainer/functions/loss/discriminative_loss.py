from chainer import cuda
from chainer.functions.activation.relu import relu
from chainer.functions.array.broadcast import broadcast_to
from chainer.functions.array.expand_dims import expand_dims
from chainer.functions.math.basic_math import absolute
from chainer.functions.math.sum import sum as c_sum


class DiscriminativeMarginBasedClusteringLoss(object):

    """Discriminative margin-based clustering loss function

    This is the implementation of the following paper:
    https://arxiv.org/abs/1708.02551

    In segmentation, one of the biggest problem is to have noise at the output
    of a trained network.
    For cross-entropy based approaches, if the pixel value is wrong,
    the loss value will be same independent from the wrong pixel's location.
    Even though the network gives wrong pixel output, it is desirable
    to have it as close as possible to the original position.
    By applying a discriminative loss function,
    groups of segmentation instances can be moved together.
    This loss function calculates the following three parameters:

    - Variance Loss:
        Loss to penalize distances between pixels which are belonging
        to the same instance. (Pull force)
    - Distance loss:
        Loss to penalize distances between the centers of instances.
        (Push force)
    - Regularization loss:
        Small regularization loss to penalize weights against overfitting.

    Args:
        delta_v (float): Minimum distance to start penalizing variance
        delta_d (float): Maximum distance to stop penalizing distance
        max_n_clusters (int): Maximum possible number of clusters.
        norm (int): Norm to calculate pixels and cluster center distances
        alpha (float): Weight for variance loss    (alpha * variance_loss)
        beta (float): Weight for distance loss     (beta * distance_loss)
        gamma (float): Weight for regularizer loss (gamma * regularizer_loss)
    """

    def __init__(self,
                 delta_v=0.5, delta_d=1.5,
                 max_n_clusters=10, norm=1,
                 alpha=1.0, beta=1.0, gamma=0.001):
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_n_clusters = max_n_clusters

        if self.max_n_clusters <= 0:
            raise ValueError("Max number of clusters has to be positive!")

        # L1 or L2 norm is allowed only
        if norm == 1:
            self.norm = lambda x, axis=None: c_sum(absolute(x), axis=axis)
        elif norm == 2:
            self.norm = lambda x, axis=None: c_sum(x ** 2, axis=axis)
        else:
            raise ValueError("For discriminative loss, "
                             "norm can only be 1 or 2. "
                             "Obtained the value : {}".format(norm))

    def __call__(self, prediction, labels):
        """Applies discriminative margin based clustering loss

        The execution steps are:

        - Calculate means
        - Calculate variance term
        - Calculate distance term
        - Calculate regularization term
        - Add weights to all and return loss value

        Args:
            prediction(:class:`~chainer.Variable` or \
                       :class:`numpy.ndarray` or \
                       :class:`cupy.ndarray`) :
                       predicted embedding vectors
                       (batch size, max cluster count, height, width)
            labels(:class:`numpy.ndarray` or \
                   :class:`cupy.ndarray`) :
                   instance segmentation ground truth
                   each unique value has to be denoting one instance
                   (batch size, height, width)

        Returns:
            Variance Loss(:class:`~chainer.Variable` or \
                          :class:`numpy.ndarray` or \
                          :class:`cupy.ndarray`):
                          variance loss multiplied by alpha
            Distance Loss(:class:`~chainer.Variable` or \
                          :class:`numpy.ndarray` or \
                          :class:`cupy.ndarray`):
                          distance loss multiplied by beta
            Regularization Loss(:class:`~chainer.Variable` or \
                                :class:`numpy.ndarray` or \
                                :class:`cupy.ndarray`):
                                regularization loss multiplied by gamma
        """

        assert(self.max_n_clusters == prediction.shape[1])

        l_var = 0.0
        l_dist = 0.0
        l_reg = 0.0
        count = 0
        means = []
        xp = cuda.get_array_module(prediction)

        labels = xp.asarray(labels, dtype=prediction.dtype)

        # Calculate cluster means
        for c in range(self.max_n_clusters):
            # Create mask for instance
            mask = xp.expand_dims(labels == c + 1, 1)
            pred_instance = prediction * xp.broadcast_to(mask, prediction.shape)

            # Calculate the number of pixels belonging to instance c
            nc = xp.asarray(xp.sum(mask, (1, 2, 3)), dtype=pred_instance.dtype)
            mean = c_sum(pred_instance, axis=(2, 3)) / \
                   xp.expand_dims(xp.maximum(nc, 1), 1)
            means.append(mean)

            # Calculate variance term
            xi = broadcast_to(expand_dims(expand_dims(mean, 2), 3),
                              pred_instance.shape) * mask
            dist = relu(self.norm(xi - pred_instance, 1) - self.delta_v) ** 2
            l_var += c_sum(dist) / xp.maximum(xp.sum(nc), 1.0)

            # Calculate regularization term
            l_reg += self.norm(mean)

        # Normalize loss by batch and instance count
        l_var /= self.max_n_clusters * prediction.shape[0]
        l_reg /= self.max_n_clusters * prediction.shape[0]

        # Calculate distance loss
        for c_a in range(self.max_n_clusters):
            for c_b in range(c_a + 1, self.max_n_clusters):
                m_a = means[c_a]
                m_b = means[c_b]
                dist = self.norm(m_a - m_b, 1)  # N
                l_dist += c_sum((relu(2 * self.delta_d - dist)) ** 2)
                count += 1
        l_dist /= count * prediction.shape[0]

        return self.alpha * l_var, self.beta * l_dist, self.gamma * l_reg


def discriminative_margin_based_clustering_loss(
        prediction, labels,
        delta_v, delta_d, max_n_clusters,
        norm=1, alpha=1.0, beta=1.0, gamma=0.001):
    """Discriminative margin-based clustering loss function

    This is the implementation of the following paper:
    https://arxiv.org/abs/1708.02551

    In segmentation, one of the biggest problem is to have noise at the output
    of a trained network.
    For cross-entropy based approaches, if the pixel value is wrong,
    the loss value will be same independent from the wrong pixel's location.
    Even though the network gives wrong pixel output, it is desirable
    to have it as close as possible to the original position.
    By applying a discriminative loss function,
    groups of segmentation instances can be moved together.
    This loss function calculates the following three parameters:

    - Variance Loss:
        Loss to penalize distances between pixels which are belonging
        to the same instance. (Pull force)
    - Distance loss:
        Loss to penalize distances between the centers of instances.
        (Push force)
    - Regularization loss:
        Small regularization loss to penalize weights against overfitting.

    Args:
        prediction(:class:`~chainer.Variable` or \
                   :class:`numpy.ndarray` or \
                   :class:`cupy.ndarray`) :
                   predicted embedding vectors
                   (batch size, max cluster count, height, width)
        labels(:class:`numpy.ndarray` or \
               :class:`cupy.ndarray`) :
               instance segmentation ground truth
               each unique value has to be denoting one instance
               (batch size, height, width)
        delta_v (float): Minimum distance to start penalizing variance
        delta_d (float): Maximum distance to stop penalizing distance
        max_n_clusters (int): Maximum possible number of clusters.
        norm (int): Norm to calculate pixels and cluster center distances
        alpha (float): Weight for variance loss    (alpha * variance_loss)
        beta (float): Weight for distance loss     (beta * distance_loss)
        gamma (float): Weight for regularizer loss (gamma * regularizer_loss)

    Returns:
        Variance Loss(:class:`~chainer.Variable` or \
                      :class:`numpy.ndarray` or \
                      :class:`cupy.ndarray`):
                      variance loss multiplied by alpha
        Distance Loss(:class:`~chainer.Variable` or \
                      :class:`numpy.ndarray` or \
                      :class:`cupy.ndarray`):
                      distance loss multiplied by beta
        Regularization Loss(:class:`~chainer.Variable` or \
                            :class:`numpy.ndarray` or \
                            :class:`cupy.ndarray`):
                            regularization loss multiplied by gamma
    """
    loss = DiscriminativeMarginBasedClusteringLoss(
        delta_v, delta_d, max_n_clusters, norm, alpha, beta, gamma)
    return loss(prediction, labels)