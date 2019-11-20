from abc import ABCMeta
from abc import abstractmethod
import contextlib
import six
import warnings


class CommunicatorBase(six.with_metaclass(ABCMeta)):
    '''Interface definition of all communicators.

    All communicators that have compatible set of methods with this
    class is supposed to work in ChainerMN's parallel computation
    implementation. The methods are named after MPI functions, such
    as ``bcast()`` came from ``MPI_Bcast()``.

    There are two types of methods: one that treats Python objects
    have ``_obj`` suffix.  The other has methods without any suffix
    and it handles ndarray and arrays filled with scaler values.  So
    the number of methods would be ::

        [send, recv, bcast, gather, allreduce] * [ '_obj', '']


    (with single exception ``alltoall``, ``multi_node_mean_grad``, ``split``
    and ``bcast_data`` so far). Also methods are supposed to be
    written in this order. All those methods must be implemented in
    its implementation class, or otherwise it cannot be instantiated
    in runtime.

    .. note:: As most implementation of ``_obj``-sufficed methods
      involves Python object pickling and unpickling, there is an
      implicit size limit.

    TODO(kuenishi): as of now no implementation class actually has
    ``allreduce`` method.

    '''
    _configs = {}

    def __init__(self):
        self._within_config_scope = False

    @property
    def rank(self):
        '''Rank (process id in the cluster) of this process in integer.'''
        raise NotImplementedError()

    @property
    def size(self):
        '''Number of processes of the cluster.'''
        raise NotImplementedError()

    @property
    def intra_rank(self):
        '''Intra rank (process id in the machine) of this process.'''
        raise NotImplementedError()

    @property
    def intra_size(self):
        '''Number of processes in the machine of this process.'''
        raise NotImplementedError()

    @property
    def inter_rank(self):
        '''The rank of this node in the cluster.'''
        raise NotImplementedError()

    @property
    def inter_size(self):
        '''Number of nodes that participates the cluster.'''
        raise NotImplementedError()

    def set_config(self, name, **kwargs):
        '''Set configurations(s) on/off

        The usage of configurations depends on each communicator. See
        :meth:`~chainermn.create_communicator` for available
        configurations.

        Args:
            name (str):
                Name of configuration to set.
            value:
                Give arbitrary object to set.
            kwargs:
                Arbitrary arguments depending on each configuration.

        '''
        raise ValueError('Unknown config: {}'.format(name))

    def get_config(self, name=None):
        '''Get configuration value(s)

        Args:
            name (str):
                Name of the configuration to get. If it is ``None``,
                all config names and values are returned.

        Returns:
            Actual value of the configuration if it is on. ``None`` if it
            is off. If ``None`` is given as ``name``, ``None`` or
            dictionary of names and configuration values is returned.

        '''
        if name is not None:
            return self._configs[name]
        return self._configs

    @abstractmethod
    def split(self, color, key):
        """A function anologous to ``MPI_Comm_Split`` .

        This method splits the inter MPI commnicator and return a wrapped
        ChainerMN communicator.

        Args:
            color (int):
                Index of new group. The process with the same color will be
                assigned to the same group.
            key (int):
                Control of rank assignment. The process will be assigned
                a rank in the new group ordered by the value of key.
                If you do not care of the rank, you can just simply specify
                the original rank.

        Returns:
            CommunicatorBase

        """
        raise NotImplementedError()

    @abstractmethod
    def alltoall(self, xs):
        '''All-to-all implementation for ndarray

        Args:
            xs (tuple of numpy/cupy array)

        Returns:
            ys (tuple of numpy/cupy array):
                Received arrays. The length of tuple equals to
                the communicator size.

        '''
        raise NotImplementedError()

    # on ndarrays and such
    @abstractmethod
    def send(self, data, dest, tag):
        '''Sends an ndarray to destination

        Receiver must invoke ``recv()`` to wait for the message.

        Args:
            data: data to be sent (tuple, list or raw numpy/cupy array)
            dest (int): Rank of the destination process
            tag (int): The tag to identify the message

        '''
        raise NotImplementedError()

    @abstractmethod
    def recv(self, source, tag):
        '''Receives an ndarray from source.

        To receive the message, sender must send the data.

        Args:
            source (int): Rank of the source process
            tag (int): The tag to specifically receive the message

        Returns:
            The data sent from source process

        '''
        raise NotImplementedError()

    @abstractmethod
    def bcast(self, data, max_buf_len=None, root=0):
        '''Broadcasts an ndarray from root process to all processes

        Args:
            data (numpy/cupy array): for root process, the data to broadcast.
                For non-root processes, this argument is ignored.
            max_buf_len (int): Length of send buffer.
            root (int): the process who has the data to broadcast.

        Returns:
            ys (numpy/cupy array) : The data sent from root process

        '''
        raise NotImplementedError()

    @abstractmethod
    def gather(self, data, root=0):
        '''Gathers an ndarray from all processes to root process

        Args:
            data (ndarray, or scaler): for root process this is ignored. For
                For non-root processes, the data to send to root process.
            root (int): rank of the process who receives the data.

        Returns:
            For root process, the ndarray sent from non-root processes.
            For non-root processes, what?

        '''
        raise NotImplementedError()

    @abstractmethod
    def allgather(self, x):
        """A primitive of inter-process all-gather communication.

        This method tries to invoke all-gather communication within the
        communicator. All processes in the communicator are expected to
        invoke ``allgather()``. This method relies on mpi4py fast communication
        optimized for numpy arrays, as well as ``send()`` and ``recv()``.

        Note that this method can only handle the same shapes of data
        over all processes, and cannot handle tuple data.

        Args:
            x (numpy/cupy array): Array to be gathered.

        Returns:
            ys (tuple of numpy/cupy array): Received arrays.
        """
        raise NotImplementedError()

    @abstractmethod
    def allreduce(self, data):
        '''Allreduce operation among processes

        Processes one of several aggregation operations using all data from
        all processes and returns the result of the aggregation to all
        processes.

        TODO(kuenishi): add ``op`` argument once we find a use case
        for operations other than 'SUM'.

        Args:
            data (ndarray): the data to aggregate among all nodes.

        Returns:
            Sum of all data from all processes.

        '''
        raise NotImplementedError()

    @abstractmethod
    def scatter(self, xs, root=0):
        """A primitive of inter-process scatter communication.

        This method tries to invoke scatter communication within the
        communicator. All processes in the communicator are expected to
        invoke ``scatter()``.

        Args:
            xs (tuple of numpy/cupy array): Arrays to be scattered.
            root (int): Rank of root process.
        Returns:
            ys (numpy/cupy array): Received arrays.
        """
        raise NotImplementedError()

    def finalize(self):
        """Finalizes and cleans up internal resource.

        The communicator SHALL NOT be used after calling this ``finalize()``.
        The behaviour is undefined when calling ``finalize`` on the same
        communicator multiple times.

        """
        pass

    # on objects
    @abstractmethod
    def send_obj(self, obj, dest, tag):
        '''Sends an arbitrary Python object to destination with a tag.

        Args:
            obj: Arbitrary object to send to receiver.
            dest (int): Rank number of receiver process (destination).
            tag: tag to identify the message.

        '''
        raise NotImplementedError()

    @abstractmethod
    def recv_obj(self, source, tag):
        '''Receives an arbitrary Python object from source process with a tag.

        Args:
           source (int): Rank number of sender process, to selectively receive
               the object.
           tag: tag to identify the message.

        Returns:
           an object sent from the source by ``send_obj``.

        '''
        raise NotImplementedError()

    @abstractmethod
    def bcast_obj(self, obj, max_buf_len=None, root=0):
        '''Broadcasts an arbitrary object from root to all non-root processes.

        Args:
            obj: arbitrary object to broadcast to all other non-root processes.
                Will be ignored at all non-root processes.
            max_buf_len (int): max length of the send buffer
            root (int): rank of the root processes who sends an object

        Returns:
            an object sent from the root process.

        '''
        raise NotImplementedError()

    @abstractmethod
    def gather_obj(self, obj, root=0):
        '''Gathers arbitrary objects from all non-root processes to the root.

        Args:
            obj: arbtrary object to send to root process. Root process will
                receive this argument included in returned list.
            root (int): rank of the root node who receives all objects.

        Returns:
            A list of objects sent from all processes.

        TODO(kuenishi): make sure the ordering of objects in the returned list.

        '''
        raise NotImplementedError()

    @abstractmethod
    def allreduce_obj(self, obj):
        '''Apply a reduce operation to all objects and spread the result.

        For example of integers and summation, equivalent local code is::

          >>> from functools import reduce
          >>> reduce(lambda x, y: x + y, [1, 2, 3, 4, 5])
          15

        The only operation currently supported is summation.

        TODO(kuenishi): support other operations such as 'MAX', 'MIN'
        and 'PROD' with ``op`` argument once we need any of them.

        Args:
           obj: An arbitrary object to apply reduce operation. Must have
               corresponding operation method e.g. ``__plus__()``.

        Returns:
           The result of the operation applied to all objects.

        '''
        raise NotImplementedError()

    # Special communication methods on grads and data of models
    @abstractmethod
    def bcast_data(self, model):
        '''Broadcast Chainer model parameter data'''
        raise NotImplementedError()

    def broadcast_data(self, model):
        '''Broadcast Chainer model parameter data

        Left for backward compatibility, but ill be deprecated in
        future version. Use ``bcast_data()`` method instad.

        '''
        self.bcast_data(model)

    @abstractmethod
    def multi_node_mean_grad(self, model, zero_fill=False):
        '''mean Chainer model gradients.

        Args:
            link (~chainer.Link): Link object.
            zero_fill: A knob to control whether to fill gradients of
              initialized and unused Link (which is None internally) with
              zero-valued array, because the all gradients must be an array
              among processes for performing all-reduce, which might be an
              array or None after backward computation. Gradients of
              uninitialized Link are skipped. If it is False, gradients of
              unused Link are just skipped.

        '''
        raise NotImplementedError()

    def allreduce_grad(self, model, zero_fill=False):
        '''mean Chainer model gradients.

        .. deprecated:: v7.0.0
            This API is deprecated. Please use
            :func:`~chainermn.CommunicatorBase.multi_node_mean_grad` instead.

        Args:
            link (~chainer.Link): Link object.
            zero_fill: A knob to control whether to fill gradients of
              initialized and unused Link (which is None internally) with
              zero-valued array, because the all gradients must be an array
              among processes for performing all-reduce, which might be an
              array or None after backward computation. Gradients of
              uninitialized Link are skipped. If it is False, gradients of
              unused Link are just skipped.

        '''
        warnings.warn('allreduce_grad() is deprecated.',
                      DeprecationWarning)
        self.multi_node_mean_grad(model, zero_fill)

    @property
    def within_config_scope(self) -> bool:
        """True if the current code is inside of an initialization scope.

        See :meth:`init_scope` for the details of the initialization scope.

        """
        return getattr(self, '_within_config_scope', False)

    @contextlib.contextmanager
    def config_scope(self):
        """Creates an configuration scope.

        """

        old_flag = self.within_config_scope
        self._within_config_scope = True
        try:
            yield
        finally:
            self._within_config_scope = old_flag

    def __setattr__(self, name, value):
        if self.within_config_scope:
            self._configs[name] = value
        super(CommunicatorBase, self).__setattr__(name, value)
