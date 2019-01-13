import chainer
from chainer import backend


class AllGather(chainer.Function):
    """Collective all-gather communication."""

    def __init__(self, comm):
        chainer.utils.experimental('chainermn.functions.AllGather')
        self.comm = comm

    def forward(self, inputs):
        x, = inputs
        return self.comm.allgather(x)

    def backward(self, inputs, grad_outputs):
        xp = backend.get_array_module(*inputs)
        gxs = self.comm.alltoall(grad_outputs)
        gx = xp.stack(gxs).sum(axis=0)
        return gx,


class AllToAll(chainer.Function):
    """Collective all-to-all communication."""

    def __init__(self, comm):
        chainer.utils.experimental('chainermn.functions.AllToAll')
        self.comm = comm

    def forward(self, inputs):
        if len(inputs) != self.comm.size:
            raise ValueError(
                'The length of inputs must be same as communicator size.')

        xs = tuple([x for x in inputs])
        return self.comm.alltoall(xs)

    def backward(self, inputs, grad_outputs):
        assert self.comm.size == len(grad_outputs)

        gys = tuple([gy for gy in grad_outputs])
        return self.comm.alltoall(gys)


class Bcast(chainer.Function):
    """Collective broadcast communication."""

    def __init__(self, comm, root):
        chainer.utils.experimental('chainermn.functions.Bcast')
        self.comm = comm
        self.root = root

    def __call__(self, *inputs):
        xp = backend.get_array_module(*inputs)

        if inputs == ():
            # Without dummy variable, this function does not "require_grad",
            # thus back propagation will not be invoked.
            dummy_var = chainer.Variable(xp.array([], dtype=xp.float32))
            dummy_var.name = 'dummy_var'
            return super(Bcast, self).__call__(dummy_var)

        else:
            return super(Bcast, self).__call__(*inputs)

    def forward(self, inputs):
        if self.comm.rank == self.root:
            x, = inputs
        else:
            x = None
        return self.comm.bcast(x, self.root),

    def backward(self, inputs, grad_outputs):
        gx, = grad_outputs
        gxs = self.comm.gather(gx, self.root)

        if self.comm.rank == self.root:
            xp = backend.get_array_module(*gxs)
            gxs = xp.stack(gxs)
            return gxs.sum(axis=0),
        else:
            return None,


class Gather(chainer.Function):
    """Collective gather communication."""

    def __init__(self, comm, root):
        chainer.utils.experimental('chainermn.functions.Gather')
        self.comm = comm
        self.root = root

    def forward(self, inputs):
        xp = backend.get_array_module(*inputs)
        x, = inputs
        ys = self.comm.gather(x, self.root)

        if self.comm.rank == self.root:
            return ys

        else:
            # Return an empty variable, which serves as "delegate_variable."
            return xp.array([], dtype=xp.float32),

    def backward(self, inputs, grad_outputs):
        return self.comm.scatter(grad_outputs, self.root),


class Scatter(chainer.Function):
    """Collective scatter communication."""

    def __init__(self, comm, root):
        chainer.utils.experimental('chainermn.functions.Scatter')
        self.comm = comm
        self.root = root

    def __call__(self, *inputs):
        xp = backend.get_array_module(*inputs)

        if inputs == ():
            # Without dummy variable, this function does not "require_grad",
            # thus back propagation will not be invoked.
            dummy_var = chainer.Variable(xp.array([], dtype=xp.float32))
            dummy_var.name = 'dummy_var'
            return super(Scatter, self).__call__(dummy_var)

        else:
            return super(Scatter, self).__call__(*inputs)

    def forward(self, inputs):
        if self.comm.rank == self.root:
            y = self.comm.scatter(inputs, self.root)
        else:
            y = self.comm.scatter(None, self.root)

        return y,

    def backward(self, inputs, grad_outputs):
        xp = backend.get_array_module(*inputs)
        gy, = grad_outputs
        gxs = self.comm.gather(gy, self.root)

        if self.comm.rank == self.root:
            return gxs

        else:
            # Slave processes need to maintain input/output shapes.
            if inputs == ():
                dummy_var = tuple([xp.array([], dtype=xp.float32)])
            else:
                dummy_var = tuple([xp.zeros(x.shape, dtype=xp.float32)
                                   for x in inputs])
            return dummy_var


def allgather(comm, x):
    """Differentiable all-gather communication between workers.

    This function invokes gather communications among processes specified
    by the communicator. Backward will be invoked as well as the ordinary
    chainer functions, where gradients are reduced to each process.

    The received array will be on the current CUDA device on the invoking
    process if ``x`` is on GPU. Please be aware that the current CUDA device
    is intended one.
    (``https://docs-cupy.chainer.org/en/stable/tutorial/basic.html#current-device``)

    Args:
        comm: ChainerMN communicator.
        x (chainer.Variables): Variables to send.

    Returns:
        ys (list of chainer.Variables): Received variables.
    """
    chainer.utils.experimental('chainermn.functions.all_gather')

    return AllGather(comm)(x)


def alltoall(comm, xs):
    """Differentiable all-to-all communication between workers.

    This function invokes all-to-all communications among processes specified
    by the communicator. Backward will be invoked as well as the ordinary
    chainer functions, just passing input gradients back.
    Unlike point-to-point communication such as ``chainermn.functions.send``
    and ``chainermn.functions.recv``, users need not to care about
    delegate variables, since ``backward()`` will not be invoked until
    all gradients from output direction arrive.
    Please refer to ``chainermn.functions.pseudo_connect`` about the detail
    of delegate variables.

    The received array will be on the current CUDA device on the invoking
    process if ``xs`` is on GPU. Please be aware that the current CUDA device
    is intended one.
    (``https://docs-cupy.chainer.org/en/stable/tutorial/basic.html#current-device``)

    Args:
        comm: ChainerMN communicator.
        xs (list of chainer.Variables): Variables to send.

    Returns:
        ys (list of chainer.Variables): Received variables.
    """
    chainer.utils.experimental('chainermn.functions.all_to_all')

    if len(xs) != comm.size:
        raise ValueError('The length of xs must be same as communicator size.')

    return AllToAll(comm)(*xs)


def bcast(comm, x, root=0):
    """Differentiable broadcast communication between workers.

    This function invokes broadcast communications among processes specified
    by the communicator. Backward will be invoked as well as the ordinary
    chainer functions, where gradients are gathered to the root process
    and summed up.

    The received array will be on the current CUDA device if ``x`` on the
    invoking process is on GPU. Please be aware that the current CUDA device
    is intended one.
    (``https://docs-cupy.chainer.org/en/stable/tutorial/basic.html#current-device``)

    Args:
        comm: ChainerMN communicator.
        x (chainer.Variable): Variable to be sent.

    Returns:
        y (chainer.Variable): Broadcasted variable.
    """
    chainer.utils.experimental('chainermn.functions.bcast')

    if comm.rank == root:
        return Bcast(comm, root)(x)
    else:
        return Bcast(comm, root)()


def gather(comm, x, root=0):
    """Differentiable gather communication between workers.

    This function invokes gather communications among processes specified
    by the communicator. Backward will be invoked as well as the ordinary
    chainer functions, where gradients are scattered from the root process
    to each slave.

    The received array will be on the current CUDA device if ``x`` on the
    root process is on GPU. Please be aware that the current CUDA device
    is intended one.
    (``https://docs-cupy.chainer.org/en/stable/tutorial/basic.html#current-device``)

    Args:
        comm: ChainerMN communicator.
        x (chainer.Variable): Variable to be sent.

    Returns:
        ys (chainer.Variable):
            Gathered variables. ``None`` for slaves.
    """
    chainer.utils.experimental('chainermn.functions.gather')

    return Gather(comm, root)(x)


def scatter(comm, xs, root=0):
    """Differentiable scatter communication between workers.

    This function invokes scatter communications among processes specified
    by the communicator. Backward will be invoked as well as the ordinary
    chainer functions, where gradients are gathered to the root process.

    The received array will be on the current CUDA device if ``xs`` on the
    root process is on GPU. Please be aware that the current CUDA device
    is intended one.
    (``https://docs-cupy.chainer.org/en/stable/tutorial/basic.html#current-device``)

    Args:
        comm: ChainerMN communicator.
        xs (list of chainer.Variable):
            Variables to be scattered for master process.
            ``None`` for slave process.

    Returns:
        y (chainer.Variable): Scattered variable.
    """
    chainer.utils.experimental('chainermn.functions.scatter')

    if comm.rank == root:
        return Scatter(comm, root)(*xs)
    else:
        return Scatter(comm, root)()
