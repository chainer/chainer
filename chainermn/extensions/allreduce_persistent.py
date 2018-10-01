import chainer
import chainer.training.extension
import numpy as np


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
        if hasattr(comm, 'mpi_comm'):
            comm = comm.mpi_comm
        else:
            # TODO(kuenishi): wrap this speciall allreduce with
            # CommunicatorBase interface
            raise ValueError(
                'allreduce_persistent is only in MPI-based communicator.')

        self.model = model
        self.comm = comm

    def __call__(self, trainer=None):
        # We need to delay MPI4py import. Please also note that _memory_utility
        # module also imports MPI4py.
        from chainermn.communicators._memory_utility \
            import array_to_buffer_object
        import mpi4py.MPI

        for _, param in sorted(_namedpersistents(self.model)):
            if hasattr(param, 'dtype') and param.dtype == np.float32:
                buf = array_to_buffer_object(param)
                self.comm.Allreduce(mpi4py.MPI.IN_PLACE, buf)
                param /= self.comm.size
            else:
                pass  # Integer persistent variables are ignored
