from chainermn.communicators.communicator_base import CommunicatorBase  # NOQA


def create_communicator(
        communicator_name='pure_nccl', mpi_comm=None,
        allreduce_grad_dtype=None, batched_copy=False):
    """Create a ChainerMN communicator.

    Different communicators provide different approaches of communication, so
    they have different performance charasteristics. The default communicator
    ``pure_nccl`` is expected to generally perform well on a variety of
    environments, so one need not to change communicators in most cases.
    However, you may need to choose other communicators depending on
    your computing platform and the availability of NCCL library.
    The following communicators are available.

    +---------------+---+---+--------+--------------------------------------+
    |Name           |CPU|GPU|NCCL    |Recommended Use Cases                 |
    +===============+===+===+========+======================================+
    |pure_nccl      |   |OK |Required|``pure_nccl`` is recommended when     |
    |               |   |   |(>= v2) |NCCL2 is available in the environment.|
    +---------------+---+---+--------+--------------------------------------+
    |flat           |   |OK |        |N/A                                   |
    +---------------+---+---+--------+--------------------------------------+
    |naive          |OK |OK |        |Testing on CPU mode                   |
    +---------------+---+---+--------+--------------------------------------+

    pure_nccl communicator supports multiple data types, FP32 and FP16,
    in gradient exchange. The communication data type is determined based on
    `chainer.global_config.dtype` and `allreduce_grad_dtype`.
    When `allreduce_grad_dtype` is the default value `None`,
    FP32 is used when `chainer.global_config.dtype` is `numpy.float32` and
    FP16 otherwise.
    `allreduce_grad_dtype` parameter,
    which is either `numpy.float16` or `numpy.float32`,
    overwrites the `chainer.global_config.dtype`.

    The table blow summarizes the data type selection in gradient exchange.

    +---------------------+--------------------------------------------+
    |                     |              allreduce_grad_dtype          |
    +---------------------+---------+------------------+---------------+
    | global_config.dtype | None    |   numpy.float16  | numpy.float32 |
    +=====================+=========+==================+===============+
    | chainer.mixed16     | FP16    |   FP16           | FP32          |
    +---------------------+---------+------------------+---------------+
    | numpy.float16       | FP16    |   FP16           | FP32          |
    +---------------------+---------+------------------+---------------+
    | numpy.float32       | FP32    |   FP16           | FP32          |
    +---------------------+---------+------------------+---------------+

    Other communicators, namely ``flat`` and ``naive``, support only
    float32 communication, no matter what the model is. This is due to
    MPI's limited support of float16.

    Args:
        communicator_name: The name of communicator (``naive``, ``flat``,
          or ``pure_nccl``)
        mpi_comm: MPI4py communicator
        allreduce_grad_dtype: Data type of gradient used in All-Reduce.
          If ``None``, the dtype of a model is used.

    Returns:
        ChainerMN communicator that implements methods defined in
        :class:`chainermn.CommunicatorBase`

    """

    if mpi_comm is None:
        try:
            import mpi4py.MPI
        except ImportError as e:
            raise ImportError(str(e) + ': '
                              'ChainerMN requires mpi4py for '
                              'distributed training. '
                              'Please read the Chainer official document '
                              'and setup MPI and mpi4py.')
        mpi_comm = mpi4py.MPI.COMM_WORLD

    if communicator_name != 'pure_nccl' and allreduce_grad_dtype is not None:
        raise ValueError(
            'allreduce_grad_dtype is only available'
            'at \'pure_nccl\' communicator.')
    if communicator_name != 'pure_nccl' and batched_copy:
        raise ValueError(
            'batched_copy is only available'
            'at \'pure_nccl\' communicator.')

    if communicator_name == 'naive':
        from chainermn.communicators.naive_communicator \
            import NaiveCommunicator
        return NaiveCommunicator(mpi_comm=mpi_comm)

    elif communicator_name == 'flat':
        from chainermn.communicators.flat_communicator \
            import FlatCommunicator
        return FlatCommunicator(mpi_comm=mpi_comm)

    elif communicator_name == 'non_cuda_aware':
        from chainermn.communicators.non_cuda_aware_communicator \
            import NonCudaAwareCommunicator
        return NonCudaAwareCommunicator(mpi_comm=mpi_comm)

    elif communicator_name == 'pure_nccl':
        from chainermn.communicators.pure_nccl_communicator \
            import PureNcclCommunicator
        return PureNcclCommunicator(mpi_comm=mpi_comm,
                                    allreduce_grad_dtype=allreduce_grad_dtype,
                                    batched_copy=batched_copy)

    elif communicator_name == 'dummy':
        from chainermn.communicators.dummy_communicator \
            import DummyCommunicator
        return DummyCommunicator(mpi_comm=mpi_comm)

    else:
        raise ValueError(
            'Unrecognized communicator: "{}"'.format(communicator_name))
