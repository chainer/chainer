import chainer
from chainer.backends import cuda
from chainer.functions.array.broadcast import broadcast_to
from chainer.functions.array.cast import cast
from chainer.functions.array.concat import concat
from chainer.functions.array.expand_dims import expand_dims
from chainer.functions.array.reshape import reshape
from chainer.functions.array.stack import stack
from chainer.functions.array.transpose import transpose
from chainer.functions.math.average import average
from chainer.functions.math.basic_math import absolute
from chainer.functions.math.maximum import maximum
from chainer.functions.math.sum import sum as c_sum

import warnings


class DiscriminativeMarginBasedClusteringLoss(object):

    """Discriminative margin-based clustering loss function

    This is the implementation of the following paper:
    https://arxiv.org/abs/1802.05591

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
        alpha (float): Weight for variance loss      (alpha * variance_loss)
        beta (float): Weight for distance loss       (beta * distance_loss)
        gamma (float): Weight for regularizer loss (gamma * regularizer_loss)
    """

    def __init__(self, delta_v, delta_d, max_n_clusters, norm=1, alpha=1.0,
                 beta=1.0, gamma=0.001):
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_n_clusters = max_n_clusters

        # L1 or L2 norm is allowed only
        if norm == 1:
            self.norm = self._l1_norm
        elif norm == 2:
            self.norm = self._l2_norm
        else:
            raise ValueError("Norm can only be 1 or 2")

    def _l1_norm(self, x, axis):
        return c_sum(absolute(x), axis=axis)

    def _l2_norm(self, x, axis):
        return c_sum(x ** 2, axis=axis)

    def _variance_term(self, pred, gt, means, delta_v, gt_idx):
        """Function to calculate variance term

        Args:
            pred (:class:`~chainer.Variable` or \
            :class:`numpy.ndarray` or '
            :class:`cupy.ndarray`):
                Prediction output
            gt (:class:`~chainer.Variable` or \
            :class:`numpy.ndarray` or '
            :class:`cupy.ndarray`):
                Ground truth output
            means (:class:`~chainer.Variable` or \
            :class:`numpy.ndarray` or '
            :class:`cupy.ndarray`):
                Instance means
            delta_v (float): Coefficient to decide 'pull force' power
            gt_idx (tuple or nd-array): Indexes of ground truth instances

        Returns:
            :class:`~chainer.Variable` or \
            :class:`numpy.ndarray` or \
            :class:`cupy.ndarray` : variance loss
        """

        bs, n_filters, n_loc = pred.shape
        n_instances = gt.shape[1]

        p = expand_dims(pred, axis=1)
        g = expand_dims(gt, axis=2)
        m = expand_dims(means, axis=3)

        p = broadcast_to(p, (bs, n_instances, n_filters, n_loc))
        g = broadcast_to(g, (bs, n_instances, n_filters, n_loc))
        m = broadcast_to(m, (bs, n_instances, n_filters, n_loc))

        m = cast(m, p.dtype)
        g = cast(g, p.dtype)

        xp = cuda.get_array_module(p)
        dv = xp.asarray(delta_v, p.dtype)

        _var = self.norm((p - m), 2)
        _var = maximum(xp.asarray(0.0, p.dtype),
                       _var - dv) ** 2  # Suppress inlier distance
        _var = _var * g[:, :, 0, :]

        var_term = 0.0
        for i in range(bs):
            if len(gt_idx[i]) < 1:
                warnings.warn("Warning : Empty gt_idx is found. " +
                              "Please check dataset content.")
                continue

            gt_sm = c_sum(g[i, gt_idx[i], 0])
            if gt_sm.data == 0:
                warnings.warn("Warning : Zero gt_sm is found. " +
                              "Please check dataset content.")
                continue

            var_sm = c_sum(_var[i, gt_idx[i]])
            if var_sm.data == 0:
                continue

            var_term += (var_sm / gt_sm)
        var_term /= bs

        return var_term

    def _distance_term(self, means, delta_d, n_objects):
        """Function to calculate distance term

        Args:
            means (:class:`~chainer.Variable` or \
            :class:`numpy.ndarray` or \
            :class:`cupy.ndarray` : ):
                Instance means
            delta_d (float): Coefficient to decide 'push force' power
            n_objects (nd-array): Instance count in current input

        Returns:
            :class:`~chainer.Variable` or \
            :class:`numpy.ndarray` or \
            :class:`cupy.ndarray` : distance loss
        """

        bs, n_instances, n_filters = means.shape
        m = cast(means, means.dtype)

        xp = cuda.get_array_module(means)
        dd = xp.asarray(delta_d, means.dtype)

        dist_term = 0.0
        for i in range(bs):

            if n_objects[i] <= 1:
                continue

            nobj = n_objects[i]

            # Prepare means
            m_i = expand_dims(m[i, :nobj, :], 1)
            m_1 = broadcast_to(m_i, (nobj, nobj, n_filters))
            m_2 = transpose(m_1, axes=(1, 0, 2))

            nrm = self.norm(m_1 - m_2, axis=2)
            margin = 2.0 * dd * (1.0 - xp.eye(nobj, dtype=means.dtype))

            _dist_term_sample = c_sum(
                maximum(xp.asarray(0.0, means.dtype), margin - nrm) ** 2)
            _dist_term_sample /= nobj * (nobj - 1)
            dist_term += _dist_term_sample

        dist_term /= bs

        return dist_term

    def _regularization_term(self, means, gt_idx):
        """Function to calculate regularization term

        Args:
            means (nd-array): Instance means
            gt_idx (tuple or nd-array): Indexes of ground truth instances

        Returns:
            :class:`~chainer.Variable` or \
            :class:`numpy.ndarray` or \
            :class:`cupy.ndarray` : regularization loss
        """

        bs, n_instances, n_filters = means.shape
        m = cast(means, means.dtype)

        reg_term = 0.0
        for i in range(bs):
            if len(gt_idx[i]) == 0:
                continue
            reg_term += average(self.norm(m[i, gt_idx[i], :], 1))
        reg_term /= bs

        return reg_term

    def _means(self, pred, gt, n_objects, max_n_objects, gt_idx):
        """Function to calculate cluster means

        Args:
            pred (:class:`~chainer.Variable` or \
                  :class:`numpy.ndarray` or \
                  :class:`cupy.ndarray`) : Prediction output
            gt (:class:`~chainer.Variable` or \
                :class:`numpy.ndarray` or \
                :class:`cupy.ndarray`) : Ground truth output
            n_objects (nd-array): Instance counts in current input
            max_n_objects (int): Maximum possible instance count
            gt_idx (tuple or nd-array):
                Indexes of ground truth instances

        Returns:
            tuple :
            (:class:`~chainer.Variable` or \
            :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`)  means
        """

        bs = pred.shape[0]
        n_filters = pred.shape[1]
        n_instances = gt.shape[1]
        n_loc = pred.shape[2]

        # Expand prediction and broadcast instances to 3rd axis
        p = reshape(pred, (bs, n_filters, 1, n_loc))
        p = broadcast_to(p, (bs, n_filters, n_instances, n_loc))

        # Expand ground truth to match the size but do not broadcast
        g = reshape(gt, (bs, 1, n_instances, n_loc))

        p = p * cast(g, p.dtype)
        xp = cuda.get_array_module(p)

        means = []
        for i in range(bs):

            p_item = p[i, :, gt_idx[i]]
            g_item = g[i, :, gt_idx[i]]

            p_sum = c_sum(p_item, axis=2)
            g_sum = cast(c_sum(g_item, axis=2), p_sum.dtype)
            _mean_sample = p_sum / g_sum

            n_fill_objects = max_n_objects - n_objects[i]

            if n_fill_objects != 0:
                _fill_sample = xp.zeros((n_fill_objects, n_filters),
                                        dtype=_mean_sample.dtype)
                _mean_sample = concat((_mean_sample, _fill_sample),
                                      axis=0)
            means.append(_mean_sample)
        means = stack(means)
        return means

    def _prepare_inputs(self, prediction, labels):
        """Function to preprocess inputs

        Args:
            prediction (:class:`~chainer.Variable` or \
            :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`):
                Prediction output
            labels (:class:`~chainer.Variable` or \
            :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`):
                Ground truth output

        Returns:
            (:class:`~chainer.Variable` or \
            :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`) :
                Prediction output
            (:class:`~chainer.Variable` or \
            :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`) :
                Ground truth output
        """
        # Reshape layers to prepare for processing

        p_shape = prediction.shape
        l_shape = labels.shape

        prediction = reshape(prediction, shape=(
            p_shape[0], p_shape[1], p_shape[2] * p_shape[3]))
        labels = reshape(labels, shape=(
            l_shape[0], l_shape[1], l_shape[2] * l_shape[3]))

        return prediction, labels

    def __call__(self, prediction, labels, n_objects, gt_idx):
        """Applies discriminative margin based clustering loss

        The execution steps are:

        - Reshape inputs to prepare for loss calculation
        - Calculate means
        - Calculate variance term
        - Calculate distance term
        - Calculate regularization term
        - Add weights to all and return loss value

        Args:
            prediction(:class:`~chainer.Variable` or \
                       :class:`numpy.ndarray` or \
                       :class:`cupy.ndarray`) :
                       segmentation prediction output
                       (batch size, total instance count, width, height)
            labels(:class:`~chainer.Variable` or \
                   :class:`numpy.ndarray` or \
                   :class:`cupy.ndarray`) :
                   segmentation ground truth
                   (batch size, total instance count, width, height)
            n_objects(:class:`~chainer.Variable` or \
                      :class:`numpy.ndarray` or \
                      :class:`cupy.ndarray`) :
                      number of objects in ground truth
                      (batch size, )
            gt_idx(:class:`~chainer.Variable` or \
                   :class:`numpy.ndarray` or \
                   :class:`cupy.ndarray`) :
                   indexes of non-zero ground truths
                   (batch size, variable length)

        Returns:
            (:class:`~chainer.Variable` or \
            :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`):
                (alpha * variance_loss) +
                (beta * distance_loss) +
                (gamma * regularizer_loss)
        """

        buffer = chainer.config.type_check
        chainer.config.type_check = False

        # Inputs
        prediction, labels = self._prepare_inputs(prediction, labels)

        # Calculate cluster means
        c_means = self._means(prediction, labels, n_objects,
                              self.max_n_clusters, gt_idx)

        # Calculate losses
        l_var = self._variance_term(prediction, labels, c_means, self.delta_v,
                                    gt_idx)
        l_dist = self._distance_term(c_means, self.delta_d, n_objects)
        l_reg = self._regularization_term(c_means, gt_idx)

        chainer.config.type_check = buffer
        return self.alpha * l_var + self.beta * l_dist + self.gamma * l_reg


def discriminative_margin_based_clustering_loss(
        prediction, labels, n_objects, gt_idx,
        delta_v, delta_d, max_n_clusters,
        norm=1, alpha=1.0, beta=1.0, gamma=0.001):
    """Discriminative margin-based clustering loss function

    This is the implementation of the following paper:
    https://arxiv.org/abs/1802.05591

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
                   segmentation prediction output
                   (batch size, total instance count, width, height)
        labels(:class:`~chainer.Variable` or \
               :class:`numpy.ndarray` or \
               :class:`cupy.ndarray`) :
               segmentation ground truth
               (batch size, total instance count, width, height)
        n_objects(:class:`~chainer.Variable` or \
                  :class:`numpy.ndarray` or \
                  :class:`cupy.ndarray`) :
                  number of objects in ground truth
                  (batch size, )
        gt_idx(:class:`~chainer.Variable` or \
               :class:`numpy.ndarray` or \
               :class:`cupy.ndarray`) :
               indexes of non-zero ground truths
               (batch size, variable length)
        delta_v (float): Minimum distance to start penalizing variance
        delta_d (float): Maximum distance to stop penalizing distance
        max_n_clusters (int): Maximum possible number of clusters.
        norm (int): Norm to calculate pixels and cluster center distances
        alpha (float): Weight for variance loss      (alpha * variance_loss)
        beta (float): Weight for distance loss       (beta * distance_loss)
        gamma (float): Weight for regularizer loss (gamma * regularizer_loss)

    Returns:
        (:class:`~chainer.Variable` or \
        :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            (alpha * variance_loss) +
            (beta * distance_loss) +
            (gamma * regularizer_loss)
    """
    loss = DiscriminativeMarginBasedClusteringLoss(
        delta_v, delta_d, max_n_clusters, norm, alpha, beta, gamma)
    return loss(prediction, labels, n_objects, gt_idx)
