from chainermn.communicators.communicator_base import CommunicatorBase  # NOQA


def create_communicator(
        communicator_name='hierarchical', mpi_comm=None,
        allreduce_grad_dtype=None):
    """Create a ChainerMN communicator.

    Different communicators provide different approaches of communication, so
    they have different performance charasteristics. The default communicator
    ``hierarchical`` is expected to generally perform well on a variety of
    environments, so one need not to change communicators in most cases.
    However, choosing proper communicator may give better performance.
    The following communicators are available.

    +---------------+---+---+--------+--------------------------------------+
    |Name           |CPU|GPU|NCCL    |Recommended Use Cases                 |
    +===============+===+===+========+======================================+
    |pure_nccl      |   |OK |Required|``pure_nccl`` is recommended when     |
    |               |   |   |(>= v2) |NCCL2 is available in the environment.|
    +---------------+---+---+--------+--------------------------------------+
    |hierarchical   |   |OK |Required|Each node has a single NIC or HCA     |
    +---------------+---+---+--------+--------------------------------------+
    |two_dimensional|   |OK |Required|Each node has multiple NICs or HCAs   |
    +---------------+---+---+--------+--------------------------------------+
    |single_node    |   |OK |Required|Single node with multiple GPUs        |
    +---------------+---+---+--------+--------------------------------------+
    |flat           |   |OK |        |N/A                                   |
    +---------------+---+---+--------+--------------------------------------+
    |naive          |OK |OK |        |Testing on CPU mode                   |
    +---------------+---+---+--------+--------------------------------------+

    Args:
        communicator_name: The name of communicator (``naive``, ``flat``,
          ``hierarchical``, ``two_dimensional``, ``pure_nccl``, or
          ``single_node``)
        mpi_comm: MPI4py communicator
        allreduce_grad_dtype: Data type of gradient used in All-Reduce.
          If ``None``, the dtype of a model is used.

    Returns:
        ChainerMN communicator that implements methods defined in
        :class:`chainermn.CommunicatorBase`

    """

    if mpi_comm is None:
        import mpi4py.MPI
        mpi_comm = mpi4py.MPI.COMM_WORLD

    if communicator_name != 'pure_nccl' and allreduce_grad_dtype is not None:
        raise ValueError(
            'allreduce_grad_dtype is only available'
            'at \'pure_nccl\' communicator.')

    if communicator_name == 'naive':
        from chainermn.communicators.naive_communicator \
            import NaiveCommunicator
        return NaiveCommunicator(mpi_comm=mpi_comm)

    elif communicator_name == 'flat':
        from chainermn.communicators.flat_communicator \
            import FlatCommunicator
        return FlatCommunicator(mpi_comm=mpi_comm)

    elif communicator_name == 'hierarchical':
        from chainermn.communicators.hierarchical_communicator \
            import HierarchicalCommunicator
        return HierarchicalCommunicator(mpi_comm=mpi_comm)

    elif communicator_name == 'two_dimensional':
        from chainermn.communicators.two_dimensional_communicator \
            import TwoDimensionalCommunicator
        return TwoDimensionalCommunicator(mpi_comm=mpi_comm)

    elif communicator_name == 'single_node':
        from chainermn.communicators.single_node_communicator \
            import SingleNodeCommunicator
        return SingleNodeCommunicator(mpi_comm=mpi_comm)

    elif communicator_name == 'non_cuda_aware':
        from chainermn.communicators.non_cuda_aware_communicator \
            import NonCudaAwareCommunicator
        return NonCudaAwareCommunicator(mpi_comm=mpi_comm)

    elif communicator_name == 'pure_nccl':
        from chainermn.communicators.pure_nccl_communicator \
            import PureNcclCommunicator
        return PureNcclCommunicator(mpi_comm=mpi_comm,
                                    allreduce_grad_dtype=allreduce_grad_dtype)

    elif communicator_name == 'dummy':
        from chainermn.communicators.dummy_communicator \
            import DummyCommunicator
        return DummyCommunicator(mpi_comm=mpi_comm)

    else:
        raise ValueError(
            'Unrecognized communicator: "{}"'.format(communicator_name))
