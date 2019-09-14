import copy

import chainer
import chainermn


def create_mnbn_model(link, comm, communication_backend='auto'):
    """Create a link object with MultiNodeBatchNormalization.

    Returns a copy of `link`, where BatchNormalization is replaced
    by MultiNodeBatchNormalization.

    Args:
        link: Link object
        comm: ChainerMN communicator
        communication_backend (str): ``mpi``, ``nccl`` or ``auto``. It is used
            to determine communication backend of MultiNodeBatchNormalization.
            If ``auto``, use the best communication backend for each
            communicator.

    Returns:
        Link object where BatchNormalization is replaced
        by MultiNodeBatchNormalization.

    """

    if isinstance(link, chainer.links.BatchNormalization):
        mnbn = chainermn.links.MultiNodeBatchNormalization(
            size=link.avg_mean.shape,
            comm=comm,
            decay=link.decay,
            eps=link.eps,
            dtype=link.avg_mean.dtype,
            use_gamma=hasattr(link, 'gamma'),
            use_beta=hasattr(link, 'beta'),
            communication_backend=communication_backend,
        )
        mnbn.copyparams(link)
        for name in link._persistent:
            mnbn.__dict__[name] = copy.deepcopy(link.__dict__[name])
        return mnbn
    elif isinstance(link, chainer.Chain):
        new_children = [
            (child_name, create_mnbn_model(link.__dict__[child_name], comm,
                                           communication_backend))
            for child_name in link._children
        ]
        new_link = copy.deepcopy(link)
        for name, new_child in new_children:
            new_link.__dict__[name] = new_child
        return new_link
    elif isinstance(link, chainer.Sequential):
        new_link = copy.deepcopy(link)
        for i, l in enumerate(link):
            new_l = create_mnbn_model(l, comm, communication_backend)
            new_link[i] = new_l
        return new_link
    elif isinstance(link, chainer.ChainList):
        new_children = [
            create_mnbn_model(l, comm, communication_backend) for l in link]
        new_link = copy.deepcopy(link)
        for i, new_child in enumerate(new_children):
            new_link._children[i] = new_child
        return new_link
    else:
        return copy.deepcopy(link)
