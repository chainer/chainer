import chainer
import chainermn
import copy

def create_mnbn_model(link, comm, communication_backend='auto'):
    """Returns a copy of `link`, where BatchNormalization is replaced
    by MultiNodeBatchNormalization.

    """
    if isinstance(link, chainer.links.BatchNormalization) :
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
            (child_name, create_mnbn_model(link.__dict__[child_name], comm, communication_backend))
            for child_name in link._children
        ]
        new_link = copy.deepcopy(link)
        for name, new_child in new_children:
            new_link.__dict__[name] = new_child
        return new_link
    elif isinstance(link, chainer.ChainList) or \
            isinstance(link, chainer.Sequential):
        new_children = [
            create_mnbn_model(l, comm, communication_backend) for l in link]
        new_link = copy.deepcopy(link)
        for i, new_child in enumerate(new_children):
            new_link._children[i] = new_child
        return new_link
    else:
        assert isinstance(link, chainer.Link)
        return copy.deepcopy(link)
