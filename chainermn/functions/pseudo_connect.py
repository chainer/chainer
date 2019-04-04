import chainer
from chainer import backend
import chainer.utils


class PseudoConnect(chainer.FunctionNode):
    """Connect a variable to a delegating variable."""

    def forward(self, inputs):
        self.retain_inputs((0,))
        # delegate_variable = inputs[0]
        actual_variables = inputs[1:]
        return actual_variables

    def backward(self, target_input_indexes, grad_outputs):
        delegate_variable, = self.get_retained_inputs()
        # actual_variables = inputs[1:]
        xp = backend.get_array_module(delegate_variable)

        # delegate_variable do not need backward gradients, instead sending
        # back dummy grads in order to take consistency of shapes of grads.
        grad_delegate_variable = xp.zeros_like(delegate_variable.array)

        # grad_outputs corresponds to grads of actual_variables.
        return (chainer.Variable(grad_delegate_variable),) + grad_outputs


def pseudo_connect(delegate_variable, *actual_variables):
    """Connect independent connected graph component.

    This function is implemented to return received arguments directly,
    except the first ``delegate_variable``.
    In backward computation, it returns received gradients directly,
    adding a zero grad corresponding to ``delegate_variable``.
    The detail of ``delegate_variable`` is described in the following notes.

    .. note::
        In model-parallel framework, models on each process might have many
        non-connected components. Here we call a given graph non-connected
        when multiple inter-process communications are needed for its
        computation. For example, consider the following example::

            class ConnectedGraph(chainermn.MultiNodeChainList):

                def __init__(self, comm):
                    super(ConnectedGraph, self).__init__(comm)
                    self.add_link(ConnectedGraphSub(), rank_in=3, rank_out=1)

        This model receives inputs from rank=3 process and sends its outputs
        to rank=1 process. The entire graph can be seen as one connected
        component ``ConnectedGraphSub``. Please refer the documentation of
        ``MultiNodeChainList`` for detail.

        On the other hand, see the next example::

            class NonConnectedGraph(chainermn.MultiNodeChainList):

                def __init__(self, comm):
                    super(NonConnectedGraph, self).__init__(comm)
                    self.add_link(NonConnectedGraphSubA(), \
rank_in=3, rank_out=1)
                    self.add_link(NonConnectedGraphSubB(), \
rank_in=1, rank_out=2)

        This model consists of two components: at first,
        ``NonConnectedGraphSubA`` receives inputs from rank=3 process and
        sends its outputs to rank=1 process, and then
        ``NonConnectedGraphSubB`` receives inputs from rank=1 process and
        sends its outputs to rank=2 process. Here multiple inter-process
        communications are invoked between ``NonConnectedGraphSubA`` and
        ``NonConnectedGraphSubB``, so it is regarded as non-connected.

        Such kind of non-connected models can be problematic in backward
        computation. Chainer traces back the computational graph from the
        output variable, however naive implementation of
        ``chainermn.functions.recv`` does not take any inputs rather receives
        inputs by ``MPI_Recv``, where backward path vanishes.

        To prevent this, dummy variables what we call ``delegate_variable``
        are used. In principle, ``chainermn.functions.send`` does not return
        any outputs because it sends data to the other process by ``MPI_Send``.
        However, ``chainermn.functions.send`` returns a dummy / empty variable
        in our implementation, which is called ``delegate_variable``. This
        variable does not hold any data, just used for retaining backward
        computation path. We can guarantee the backward computation just by
        putting ``delegate_variable`` to the next ``chainermn.functions.recv``
        (``chainermn.functions.recv`` has an optional argument to receive
        ``delegate_variable``).

    .. note::
        In some cases the intermediate graph component returns model outputs.
        See the next example::

            class NonConnectedGraph2(chainermn.MultiNodeChainList):

                def __init__(self, comm):
                    super(NonConnectedGraph2, self).__init__(comm)
                    self.add_link(NonConnectedGraphSubA(), \
rank_in=1, rank_out=None)
                    self.add_link(NonConnectedGraphSubB(), \
rank_in=None, rank_out=1)

        This model first receives inputs from rank=1 process and make model
        outputs (specified by ``rank_out=None``) in ``NonConnectedGraphSubA``.
        Then using model inputs (specified by ``rank_in=None``),
        ``NonConnectedGraphSubB`` sends its outputs to rank=1 process. Since
        ``MultiNodeChainList.__call__`` returns outputs of the last component
        (in this case, outputs of ``NonConnectedGraphSubB``), naive
        implementation cannot output the returned value of
        ``NonConnectedGraphSubA`` as the model outputs. In this case,
        ``pseudo_connect`` should be used.

        ``pseudo_connect`` takes two arguments. The first one
        ``delegate_variable`` is what we explained in above note. In this
        case, returned value of ``NonConnectedGraphSubB`` corresponds to
        ``delegate_variable``. The second one ``actual_variables`` is
        "what we want ``delegate_variable`` to imitate". In
        ``NonConnectedGraph2``, we obtain returned value of
        ``NonConnectedGraphSubB`` as the model outputs, but what we actually
        want is returned value of ``NonConnectedGraphSubA``. At the same time
        we want to trace back this resulted variable in backward computation.
        Using ``pseudo_connect``, we can make a variable whose data is the
        same as the returned value of ``NonConnectedGraphSubA``, and which
        traces back ``NonConnectedGraphSubB`` first.

        ``pseudo_connect`` should also be used in some pathological cases,
        for example, where multiple ``chainermn.functions.send`` occurs
        sequentially.

    Args:
        delegate_variable (chainer.Variable):
            Pointer to the previous non-connected graph component.
        actual_variables (tuple of chainer.Variable):
            Actual values which ``delegate_variable`` imitate.

    Returns:
        tuple of chainer.Variable:
            A variable with the given values combined with delegating variable.
    """
    chainer.utils.experimental('chainermn.functions.pseudo_connect')
    if delegate_variable is None:
        xp = backend.get_array_module(*actual_variables)
        delegate_variable = xp.empty((0,), xp.float32)
    return PseudoConnect().apply(
        (delegate_variable,) + actual_variables)
