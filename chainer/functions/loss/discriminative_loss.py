from chainer import backend
from chainer.functions.activation.relu import relu
from chainer.functions.array.broadcast import broadcast_to
from chainer.functions.array.stack import stack
from chainer.functions.math.average import average
from chainer.functions.math.basic_math import absolute
from chainer.functions.math.sqrt import sqrt
from chainer.functions.math.sum import sum as c_sum
from chainer.utils import argument


class DiscriminativeMarginBasedClusteringLoss(object):

    """Discriminative margin-based clustering loss function

    This is the implementation of the following paper:
    https://arxiv.org/abs/1708.02551
    It calculates pixel embeddings, and calculates three different terms
    based on those embeddings and applies them as loss.
    This loss penalizes the pixel embeddings according to following items:

    Same instance's embeddings have to be closer to each other (pull force)
    Different instance's, they have to be further away (push force).
    The loss also brings a weak regularization term to prevent overfitting.
    This loss function calculates the following three parameters:

    Variance Loss
        Loss to penalize distances between pixels which are belonging
        to the same instance. (Pull force)

    Distance loss
        Loss to penalize distances between the centers of instances.
        (Push force)

    Regularization loss
        Small regularization loss to penalize weights against overfitting.

    """

    def __init__(self, delta_v=0.5, delta_d=1.5, max_embedding_dim=None,
                 norm=1, alpha=1.0, beta=1.0, gamma=0.001, **kwargs):
        argument.parse_kwargs(kwargs)
        if max_embedding_dim is not None:
            warnings.warn(
                'max_embedding_dim argument'
                ' is not supported anymore. '
                'This information is obtained'
                ' from channel of input array',
                DeprecationWarning)

        self.delta_v = delta_v
        self.delta_d = delta_d
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # L1 or L2 norm is allowed only
        if norm == 1:
            self.norm = lambda x, axis=None: c_sum(absolute(x), axis=axis)
        elif norm == 2:
            self.norm = lambda x, axis=None: sqrt(c_sum(x ** 2, axis=axis))
        else:
            raise ValueError('For discriminative loss, '
                             'norm can only be 1 or 2. '
                             'Obtained the value : {}'.format(norm))

    def __call__(self, embeddings, labels):
        """
        Args:
            embeddings (:class:`~chainer.Variable` or :ref:`ndarray`):
                predicted embedding vectors
                (batch size, embedding dimensions, height, width)

            labels (:ref:`ndarray`):
                instance segmentation ground truth
                each unique value has to be denoting one instance.
                Each instance's value has to be positive
                (batch size, height, width)

        Returns:
            :class:`tuple` of :class:`chainer.Variable`:
            - *Variance loss*: Variance loss multiplied by ``alpha``
            - *Distance loss*: Distance loss multiplied by ``beta``
            - *Regularization loss*: Regularization loss multiplied by
              ``gamma``

        """

        shape = embeddings.shape
        assert len(shape) == 4

        b = embeddings.shape[0]

        xp = backend.get_array_module(embeddings)

        unique_label_idx = xp.unique(labels)
        # Remove background label
        unique_label_idx = unique_label_idx[unique_label_idx >= 0]

        var_loss = xp.zeros((b,))
        means = []

        # Find active labels per batch item
        active_id_count = xp.zeros((b, ), dtype=embeddings.dtype)
        active_idxs = []
        for b_idx in range(b):
            active_id = xp.unique(labels[b_idx])
            active_id = active_id[active_id >= 0]
            active_idxs.append(active_id)
            active_id_count[b_idx] = len(active_id)

        # Calculate mean embeddings and variance loss
        for idx in unique_label_idx:
            mask = labels == idx

            number_of_pixels = xp.maximum(xp.sum(mask, (1, 2)), 1)[:, None]
            instance_embeddings = embeddings * mask[:, None, :, :]

            mean_embeddings = c_sum(instance_embeddings, (2, 3))
            mean_embeddings = mean_embeddings / number_of_pixels
            means.append(mean_embeddings)

            # Variance loss
            me = broadcast_to(mean_embeddings[:, :, None, None],
                              instance_embeddings.shape)
            local_var_loss = instance_embeddings - me
            local_var_loss *= mask[:, None]
            local_var_loss = self.norm(local_var_loss, 1)
            local_var_loss = relu(local_var_loss - self.delta_v) ** 2
            local_var_loss = c_sum(local_var_loss, (1, 2))
            dividend = xp.maximum(active_id_count, 1) * number_of_pixels[:, 0]
            var_loss += local_var_loss / dividend

        var_loss = average(var_loss)

        # Calculate mean distance loss
        means = stack(means, 1)

        dist_loss = xp.asarray(0.0, dtype=embeddings.dtype)
        counter = 0
        for b_idx in range(b):
            active_ids = active_idxs[b_idx]
            for c1_idx in range(len(active_ids)):
                for c2_idx in range(c1_idx + 1, len(active_ids)):
                    m_diff = means[b_idx][c1_idx] - means[b_idx][c2_idx]
                    m_diff = self.norm(m_diff)
                    m_diff = relu(2 * self.delta_d - m_diff) ** 2
                    dist_loss = dist_loss + m_diff
                    counter += 1
        dist_loss = dist_loss / xp.maximum(counter, 1)

        # Calculate regularization term
		mx_active = xp.maximum(active_id_count, 1)
        reg_loss = average(self.norm(means, (1, 2)) / mx_active)

        rtn = (self.alpha * var_loss,
               self.beta * dist_loss,
               self.gamma * reg_loss)
        return rtn


def discriminative_margin_based_clustering_loss(
        embeddings, labels,
        delta_v, delta_d, max_embedding_dims=None,
        norm=1, alpha=1.0, beta=1.0, gamma=0.001):
    """Discriminative margin-based clustering loss function

    This is the implementation of the following paper:
    https://arxiv.org/abs/1708.02551
    It calculates pixel embeddings, and calculates three different terms
    based on those embeddings and applies them as loss.
    The main idea is that the pixel embeddings
    for same instances have to be closer to each other (pull force),
    for different instances, they have to be further away (push force).
    The loss also brings a weak regularization term to prevent overfitting.
    This loss function calculates the following three parameters:

    Variance Loss
        Loss to penalize distances between pixels which are belonging
        to the same instance. (Pull force)

    Distance loss
        Loss to penalize distances between the centers of instances.
        (Push force)

    Regularization loss
        Small regularization loss to penalize weights against overfitting.

    For the labels, any positive value, including zero will be handled as a
    separate instance. Any negative value in the ground truth will be
    will be exempt from loss calculation. This loss is designed for dense
    labels. Hence, the performance is not optimized for sparse label arrays.

    Args:
        embeddings (:class:`~chainer.Variable` or :ref:`ndarray`):
            predicted embedding vectors
            (batch size, embedding dimensions, height, width)

        labels (:ref:`ndarray`):
            instance segmentation ground truth
            each unique value has to be denoting one instance
            (batch size, height, width)
        delta_v (float): Minimum distance to start penalizing variance
        delta_d (float): Maximum distance to stop penalizing distance
        max_embedding_dims (float): Deprecated
        norm (int): Norm to calculate pixels and cluster center distances
        alpha (float): Weight for variance loss
        beta (float): Weight for distance loss
        gamma (float): Weight for regularization loss

    Returns:
        :class:`tuple` of :class:`chainer.Variable`:
        - *Variance loss*: Variance loss multiplied by ``alpha``
        - *Distance loss*: Distance loss multiplied by ``beta``
        - *Regularization loss*: Regularization loss multiplied by ``gamma``

    """

    loss = DiscriminativeMarginBasedClusteringLoss(
        delta_v, delta_d, max_embedding_dims, norm, alpha, beta, gamma)
    return loss(embeddings, labels)
