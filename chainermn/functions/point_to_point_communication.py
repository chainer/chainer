import chainer
from chainer import backend
import chainer.utils


class Send(chainer.Function):
    """Send elements to target process."""

    def __init__(self, comm, peer_rank, peer_tag):
        chainer.utils.experimental('chainermn.functions.Send')
        self.comm = comm
        self.peer_rank = peer_rank
        self.peer_tag = peer_tag

    @property
    def label(self):
        return '{} (peer_rank: {})'.format(
            self.__class__.__name__,
            self.peer_rank)

    def forward(self, inputs):
        xp = backend.get_array_module(*inputs)

        # The last input is dummy variable, to retain gradient computation
        # of this function.
        xs = inputs[:-1]

        if len(xs) == 1:
            xs = xs[0]

        self.comm.send(xs, self.peer_rank, self.peer_tag)

        # Return an empty variable, which serves as "delegate_variable."
        return xp.array([], dtype=xp.float32),

    def backward(self, inputs, grad_outputs):
        xp = backend.get_array_module(*inputs)
        dummy_grad = xp.array([], dtype=xp.float32)
        grad = self.comm.recv(self.peer_rank, self.peer_tag)
        if isinstance(grad, tuple):
            return tuple([xp.array(gy) for gy in grad] + [dummy_grad])
        else:
            return xp.array(grad), dummy_grad


class Recv(chainer.Function):
    """Receive elements from target process."""

    def __init__(self, comm, peer_rank, peer_tag):
        chainer.utils.experimental('chainermn.functions.Recv')
        self.comm = comm
        self.peer_rank = peer_rank
        self.peer_tag = peer_tag

    def __call__(self, *inputs):
        xp = backend.get_array_module(*inputs)

        if inputs == ():
            # Expected to be invoked without any args in usual case.
            dummy_var = chainer.Variable(xp.array([], dtype=xp.float32))
            dummy_var.name = 'dummy_var'
            return super(Recv, self).__call__(dummy_var)

        else:
            # Used for retaining computational graph.
            return super(Recv, self).__call__(*inputs)

    @property
    def label(self):
        return '{} (peer_rank: {})'.format(
            self.__class__.__name__,
            self.peer_rank)

    def forward(self, inputs):
        data = self.comm.recv(self.peer_rank, self.peer_tag)

        if not isinstance(data, tuple):
            data = tuple([data])

        return data

    def backward(self, inputs, grad_outputs):
        xp = backend.get_array_module(*inputs)
        self.comm.send(grad_outputs, self.peer_rank, self.peer_tag)

        # dummy_var is needed to maintain Chainer's constraint.
        if inputs == ():
            dummy_var = tuple([xp.array([], dtype=xp.float32)])
        else:
            dummy_var = tuple([xp.zeros(x.shape, dtype=xp.float32)
                               for x in inputs])

        return dummy_var


def send(x, communicator, rank, tag=0):
    """Send elements to target process.

    This function returns a dummy variable only holding the computational
    graph. If ``backward()`` is invoked by this dummy variable, it will
    try to receive gradients from the target process and send them back
    to the parent nodes.

    Args:
        x (Variable): Variable holding a matrix which you would like to send.
        communicator (chainer.communicators.CommunicatorBase):
            ChainerMN communicator.
        rank (int): Target process specifier.
        tag (int): Optional message ID (MPI feature).

    Returns:
        ~chainer.Variable:
            A dummy variable with no actual data, only holding the
            computational graph. Please refer
            ``chainermn.functions.pseudo_connect`` for detail.

    """
    chainer.utils.experimental('chainermn.functions.send')

    if rank == communicator.rank:
        raise ValueError(
            'rank must be different from communicator rank, '
            'otherwise deadlock occurs')

    xp = backend.get_array_module(*x)

    # Dummy variable to retain gradient computation of send,
    # otherwise the corresponding recv will cause deadlock in backward
    # in the case where all inputs for this function does not require_grad.
    dummy_var = chainer.Variable(xp.array([], dtype=xp.float32))

    if isinstance(x, list) or isinstance(x, tuple):
        inputs = x + type(x)([dummy_var])
        delegate_variable = Send(
            communicator, peer_rank=rank, peer_tag=tag)(*inputs)
    else:
        delegate_variable = Send(
            communicator, peer_rank=rank, peer_tag=tag)(x, dummy_var)

    delegate_variable.name = 'delegate_variable'
    return delegate_variable


def recv(communicator, rank, delegate_variable=None, tag=0, force_tuple=False):
    """Receive elements from target process.

    This function returns data received from target process. If ``backward()``
    is invoked, it will try to send gradients to the target process.
    The received array will be on the current CUDA device if the corresponding
    ``send()`` is invoked with arrays on GPU.
    Please be aware that the current CUDA device is intended one.
    (``https://docs-cupy.chainer.org/en/stable/tutorial/basic.html#current-device``)

    .. note::
        If you define non-connected computational graph on one process,
        you have to use ``delegate_variable`` to specify the output of
        previous computational graph component.
        Otherwise ``backward()`` does not work well.
        Please refer ``chainermn.functions.pseudo_connect`` for detail.

    Args:
        communicator (chainer.communicators.CommunicatorBase):
            ChainerMN communicator.
        rank (int): Target process specifier.
        delegate_variable (chainer.Variable):
            Pointer to the other non-connected component.
        tag (int): Optional message ID (MPI feature).
        force_tuple (bool): If ``False`` (the default) a Variable will be
            returned when the number of outputs is one. Otherwise, this
            method returns a tuple even when the number of outputs is one.

    Returns:
        ~chainer.Variable:
            Data received from target process. If ``backward()`` is invoked
            by this variable, it will send gradients to the target process.

    """
    chainer.utils.experimental('chainermn.functions.recv')

    if rank == communicator.rank:
        raise ValueError(
            'rank must be different from communicator rank, '
            'otherwise deadlock occurs')

    if delegate_variable is None:
        res = Recv(
            communicator,
            peer_rank=rank,
            peer_tag=tag)()
    else:
        delegate_variable.name = 'delegate_variable'
        res = Recv(
            communicator,
            peer_rank=rank,
            peer_tag=tag)(delegate_variable)

    if force_tuple and not isinstance(res, tuple):
        return tuple([res])
    else:
        return res
