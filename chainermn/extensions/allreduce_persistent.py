import chainer
import chainer.training.extension


def _namedpersistents(model):
    assert isinstance(model, chainer.Link)

    for lname, link in model.namedlinks():
        for pname in link._persistent:
            yield lname + '/' + pname, link.__dict__[pname]


class AllreducePersistent(chainer.training.extension.Extension):

    """Chainer extension to averagize persistents over workers.

    When called, this extension invokes all-reduce communication among
    workers to compute averages of persistent variables in the model.
    Persistent variables are updated to the averages. Currently, we ignore
    integer persistent variables, and only float persistent variables are
    handled.

    This extension is mainly to improve the running mean and variance of
    BatchNormalization by increasing the effective number of examples.
    We do not need to call this frequently; call just before storing or
    evaluating the model.

    Args:
        model (chainer.link.Link): Target link object.
        comm (ChainerMN communicator): communicator to compute averages.
    """

    trigger = 1, 'epoch'

    # This extension should be called earlier than evaluators.
    priority = chainer.training.extension.PRIORITY_WRITER + 1

    def __init__(self, model, comm):
        self.model = model
        self.comm = comm

    def __call__(self, trainer=None):
        for _, param in sorted(_namedpersistents(self.model)):
            if hasattr(param, 'dtype'):
                self.comm.multi_node_mean(None, param)
            else:
                pass  # Integer persistent variables are ignored
