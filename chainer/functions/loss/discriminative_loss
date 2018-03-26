
class DiscriminativeLoss:
    """ Discriminative loss function

    This is the implementation of following paper: https://arxiv.org/pdf/1802.05591.pdf

    In segmentation, one of the biggest problem is having noise at the output of trained network.
    For cross-entropy based approaches, if the pixel value is wrong,
    the loss value will be same independent from wrong pixel's location.
    However, for segmentation, even though network gives wrong pixel output, it is desirable to have it
    as close as possible to the original position.
    By applying discriminative loss function, groups of segmentation instances can be moved together.

    This loss function calculates three different parameters:
        - Variance Loss:
            Loss to penalize distances between pixels which are belonging to same instance. (Pull force)
        - Distance loss:
            Loss to penalize distances between the centers of instances. (Push force)
        - Regularizer loss:
            Small regularization loss to penalize weights against overfit

    :param delta_v: Used to define minimum distance to start penalizing variance
    :param delta_d: Used to define maximum distance to stop penalizing distance
    :param max_n_clusters: Used to define maximum possible number of clusters.
    :param norm: Norm to calculate distance between pixels and cluster centers
    :param alpha: Weight for variance loss      (alpha * variance_loss)
    :param beta: Weight for distance loss       (beta * distance_loss)
    :param gamma: Weight for regularizer loss   (gamma * regularizer_loss)

    :return loss: (alpha * variance_loss) + (beta * distance_loss) + (gamma * regularizer_loss)

    """
    def __init__(self, delta_v, delta_d, max_n_clusters, norm=1,  alpha=1.0, beta=1.0, gamma=0.001):
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_n_clusters = max_n_clusters

        # L1 or L2 norm is allowed only
        if norm != 1:
            raise Exception('Currently discriminative loss only supports L1 Norm')

        if norm == 2:
            self.norm = self.l2_norm
        else:
            self.norm = self.l1_norm

        return

    def l1_norm(self, x, axis):
        return F.sum(F.absolute(x), axis=axis)

    def l2_norm(self, x, axis):
        return F.sqrt(F.sum(x ** 2, axis=axis))

    def variance_term(self, pred, gt, means, delta_v,  gt_idx):
        """
        pred: bs, n_filters, height * width
        gt: bs, n_instances, height * width
        means: bs, n_instances, n_filters
        """

        bs, n_filters, n_loc = pred.shape
        n_instances = gt.shape[1]

        # Prepare each item with same size by broadcasting missing axes
        p = F.expand_dims(pred, axis=1)
        g = F.expand_dims(gt, axis=2)
        m = F.expand_dims(means, axis=3)

        p = F.broadcast_to(p, (bs, n_instances, n_filters, n_loc))
        g = F.broadcast_to(g, (bs, n_instances, n_filters, n_loc))
        m = F.broadcast_to(m, (bs, n_instances, n_filters, n_loc))

        m = F.cast(m, p.dtype)
        g = F.cast(g, p.dtype)
        dv = cp.asarray(delta_v, p.dtype)

        _var = self.norm((p - m), 2)
        _var = F.maximum(cp.asarray(0.0, np.float32), _var - dv) ** 2    # Suppress inlier distance
        _var = _var * g[:, :, 0, :]

        var_term = 0.0
        for i in range(bs):
            var_term += F.sum(_var[i, gt_idx[i]]) / F.sum(g[i, gt_idx[i], 0])
        var_term = var_term / cp.asarray(bs, dtype=np.float32)

        return var_term

    def distance_term(self, means, delta_d, n_objects):
        """
        means: bs, n_instances, n_filters
        """

        bs, n_instances, n_filters = means.shape

        m = F.cast(means, np.float32)
        dd = cp.asarray(delta_d, np.float32)

        dist_term = 0.0
        for i in range(bs):

            if n_objects[i] <= 1:
                continue

            nobj = n_objects[i].get()

            # Prepare means
            m_i = F.expand_dims(m[i, :nobj, :], 1)
            m_1 = F.broadcast_to(m_i, (nobj, nobj, n_filters))
            m_2 = F.transpose(m_1, axes=(1, 0, 2))

            nrm = self.norm(m_1 - m_2, axis=2)
            margin = 2.0 * dd * (1.0 - cp.eye(nobj, dtype=np.float32))

            _dist_term_sample = F.sum(F.maximum(cp.asarray(0.0, np.float32), margin - nrm) ** 2)
            dist_term += _dist_term_sample / cp.asarray((nobj * (nobj - 1)), np.float32)

        dist_term = dist_term / cp.asarray(bs, dtype=np.float32)

        return dist_term

    def regularization_term(self, means, n_objects):
        """
        means: bs, n_instances, n_filters
        """

        bs, n_instances, n_filters = means.shape
        m = F.cast(means, np.float32)

        reg_term = 0.0
        for i in range(bs):
            reg_term += F.mean(self.norm(m[i, : n_objects[i], :], 1))
        reg_term = reg_term / cp.asarray(bs, dtype=np.float32)

        return reg_term

    def means(self, pred, gt, n_objects, max_n_objects, gt_idx):
        """pred: bs, n_filters, height * width
           gt: bs, n_instances, height * width"""

        bs = pred.shape[0]
        n_filters = pred.shape[1]
        n_instances = gt.shape[1]
        n_loc = pred.shape[2]

        # Expand prediction and broadcast instances to 3rd axis
        p = F.reshape(pred, (bs, n_filters, 1, n_loc))
        p = F.broadcast_to(p, (bs, n_filters, n_instances, n_loc))

        # Expand ground truth to match the size but do not broadcast
        g = F.reshape(gt, (bs, 1, n_instances, n_loc))

        p = p * F.cast(g, p.dtype)

        means = []
        for i in range(bs):

            p_item = p[i, :, gt_idx[i]]
            g_item = g[i, :, gt_idx[i]]

            p_sum = F.sum(p_item, axis=2)
            g_sum = F.cast(F.sum(g_item, axis=2), p_sum.dtype)
            _mean_sample = p_sum / g_sum

            n_fill_objects = max_n_objects - n_objects[i]

            assert n_fill_objects >= 0

            if n_fill_objects != 0:
                _fill_sample = cp.zeros((cp.asnumpy(n_fill_objects), n_filters), dtype=_mean_sample.dtype)
                _mean_sample = F.concat((_mean_sample, _fill_sample), axis=0)

            means.append(_mean_sample)

        means = F.stack(means)
        return means

    def prepare_inputs(self, prediction, labels):
        # Reshape layers to prepare for processing
        p_shape = prediction.shape
        l_shape = labels.shape

        prediction = F.reshape(prediction, shape=(p_shape[0], p_shape[1], p_shape[2] * p_shape[3]))
        labels = F.reshape(labels, shape=(l_shape[0], l_shape[1], l_shape[2] * l_shape[3]))

        return prediction, labels

    def apply(self, *x):
        """
        :param x:
        x[0] = segmentation prediction output
        x[1] = segmentation ground truth
        x[2] = number of objects in ground truth
        x[3] = indexes of non-zero ground truths
        :return:
        Loss value
        """

        assert len(x) == 4
        assert len(x[0].shape) == 4
        assert len(x[1].shape) == 4

        # Inputs
        prediction, labels = self.prepare_inputs(x[0], x[1])
        n_objects, gt_idx = x[2:4]

        # Calculate cluster means
        c_means = self.means(prediction, labels, n_objects, self.max_n_clusters, gt_idx)

        # Calculate losses
        l_var = self.variance_term(prediction, labels, c_means, self.delta_v, gt_idx)
        l_dist = self.distance_term(c_means, self.delta_d, n_objects)
        l_reg = self.regularization_term(c_means, n_objects)

        return self.alpha * l_var + self.beta * l_dist + self.gamma * l_reg

    def __call__(self, *args):
        return self.apply(*args)
