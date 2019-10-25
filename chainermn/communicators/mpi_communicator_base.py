import mpi4py
import numpy

import chainer
import chainer.backends
import chainer.utils
from chainer.utils import collections_abc
from chainermn.communicators import _communication_utility
from chainermn.communicators._communication_utility import chunked_bcast_obj
from chainermn.communicators import _memory_utility
from chainermn.communicators import communicator_base
import chainerx


_dtype_mpi_type = {
    # see the definition of mpi4py.MPI._typedict (in mpi4py/MPI/typemap.pxi)
    numpy.dtype(numpy.int32): mpi4py.MPI._typedict['i'],
    numpy.dtype(numpy.int64): mpi4py.MPI._typedict['l'],
    numpy.dtype(numpy.float16): mpi4py.MPI._typedict['f'],
    numpy.dtype(numpy.float32): mpi4py.MPI._typedict['f'],
    numpy.dtype(numpy.float64): mpi4py.MPI._typedict['d'],
}


def _check_dtype(caller, msgtype):
    dtype = msgtype.dtype
    if dtype not in _dtype_mpi_type.keys():
        raise TypeError(
            '{} does not support dtype {}'.format(caller, dtype))


def _check_dtypes_are_same(msgtypes):
    dtypes = [msgtype.dtype for msgtype in msgtypes]
    if any(dtypes[0] != dtype for dtype in dtypes):
        raise TypeError('all dtypes must be the same')


def _is_numpy_array(array):
    return isinstance(array, numpy.ndarray)


def _is_cupy_array(array):
    return chainer.backend.get_array_module(array) is not numpy


def _cnt_to_dsp(cnt):
    """Utility to convert length array to cumulative array."""
    return [0] + numpy.cumsum(cnt)[:-1].tolist()


def _get_mpi_type(msgtype):
    dtype = msgtype.dtype
    if dtype not in _dtype_mpi_type.keys():
        raise TypeError(
            'dtype {} is not supported by MpiCommunicator'.format(dtype))

    return _dtype_mpi_type[dtype]


class _MessageType(object):

    def __init__(self, obj):
        if _is_numpy_array(obj) or _is_cupy_array(obj):
            self.is_host = _is_numpy_array(obj)
            self.is_tuple = False
            self.narr = 1
            self.ndims = [obj.ndim]
            self.shapes = [obj.shape]
            self.dtype = obj.dtype
        elif isinstance(obj, collections_abc.Iterable):
            if all(map(_is_numpy_array, obj)):
                self.is_host = True
            elif all(map(_is_cupy_array, obj)):
                self.is_host = False
            else:
                raise ValueError(
                    'All message objects must be either numpy or cupy arrays.')
            self.is_tuple = True
            self.narr = len(obj)
            self.ndims = [x.ndim for x in obj]
            self.shapes = [x.shape for x in obj]
            dtypes = [x.dtype for x in obj]
            if not all(dtype == dtypes[0] for dtype in dtypes):
                raise TypeError(
                    'Message objects must be the same dtype')
            self.dtype = dtypes[0]
        else:
            raise TypeError(
                'Message object must be numpy/cupy array or its tuple.')

    def get_array_module(self):
        if self.is_host:
            return numpy
        else:
            import cupy
            return cupy


class MpiCommunicatorBase(communicator_base.CommunicatorBase):
    '''MpiCommunicatorBase

    Implementation of communicator interface defined by
    :class:`CommunicatorBase`. This communicator assumes MPI4py and
    all ChainerMN processes are invoked by ``mpirun`` (``mpiexec``)
    command. Although this lacks several important methods such as
    ``multi_node_mean_grad`` to be impelmented with speficic algorithm. See
    hierarchical communicator or pure_nccl communicator for example.

    '''

    def __init__(self, mpi_comm):
        self.mpi_comm = mpi_comm
        self._init_ranks()
        with self.config_scope():
            self.batched_copy = False

    @property
    def rank(self):
        return self.mpi_comm.rank

    @property
    def size(self):
        return self.mpi_comm.size

    @property
    def intra_rank(self):
        return self._intra_rank

    @property
    def intra_size(self):
        return self._intra_size

    @property
    def inter_rank(self):
        return self._inter_rank

    @property
    def inter_size(self):
        return self._inter_size

    def set_config(self, name, value=True, **kwargs):
        if name == 'batched_copy':
            with self.config_scope():
                self.batched_copy = value
        else:
            # Although MpiCommunicatorBase has no ancestor, practice
            return super(MpiCommunicatorBase, self).set_config(name, **kwargs)

    def get_config(self, name=None):
        if name == 'batched_copy':
            return self.batched_copy
        else:
            # Although MpiCommunicatorBase has no ancestor, practice.
            return super(MpiCommunicatorBase, self).get_config(name)

    def split(self, color, key):
        return self.__class__(mpi_comm=self.mpi_comm.Split(color, key))

    def alltoall(self, xs):
        """A primitive of inter-process all-to-all function.

        This method tries to invoke all-to-all communication within the
        communicator. All processes in the communicator are expected to
        invoke ``alltoall()``. This method relies on mpi4py fast communication
        optimized for numpy arrays, as well as ``send()`` and ``recv()``.

        If ``xs`` is numpy array, the returned array will also be allocated
        as numpy array. Additionally, when ``xs`` is cupy array, the returned
        array will be placed at current device
        (``https://docs-cupy.chainer.org/en/stable/tutorial/basic.html#current-device``)
        regardless of which device the argument is placed at remote nodes.

        Args:
            xs (tuple of numpy/cupy array)

        Returns:
            ys (tuple of numpy/cupy array):
                Received arrays. The length of tuple equals to
                the communicator size.
        """
        chainer.utils.experimental(
            'chainermn.communicators.MpiCommunicatorBase.alltoall')

        if len(xs) != self.size:
            raise ValueError(
                'The length of data must be same as communicator size.')

        # Type check.
        msgtypes = [_MessageType(x) for x in xs]
        for msgtype in msgtypes:
            _check_dtype('alltoall', msgtype)
        _check_dtypes_are_same(msgtypes)
        send_msgtype = msgtypes[0]

        msgtypes = self.mpi_comm.alltoall(msgtypes)
        _check_dtypes_are_same(msgtypes)
        recv_msgtype = msgtypes[0]

        # Collective communication.
        slens = [x.size for x in xs]
        xp = chainer.backend.get_array_module(*xs)
        sbuf = xp.hstack([x.reshape(-1) for x in xs])
        shapes = [msgtype.shapes[0] for msgtype in msgtypes]
        rlens = [chainer.utils.size_of_shape(s) for s in shapes]
        rbuf = xp.empty([sum(rlens)], dtype=msgtype.dtype)
        if xp is not numpy:
            sbuf = _memory_utility.get_device_memory_pointer(sbuf)
            chainer.cuda.Stream.null.synchronize()
        self.mpi_comm.Alltoallv(
            [sbuf, (slens, _cnt_to_dsp(slens)), _get_mpi_type(send_msgtype)],
            [_memory_utility.get_device_memory_pointer(rbuf),
             (rlens, _cnt_to_dsp(rlens)), _get_mpi_type(recv_msgtype)])
        ys = [rbuf[i:i + l].reshape(s)
              for i, l, s in zip(_cnt_to_dsp(rlens), rlens, shapes)]

        return tuple(ys)

    def send(self, data, dest, tag):
        """A primitive for inter-process transmitter.

        This method sends numpy-array to target process.
        The target process is expected to invoke ``recv()``.
        This method relies on mpi4py fast communication optimized for
        numpy arrays, which discards any information attached to
        chainer.Variable objects. Please be sure.

        Args:
            data: data to be sent (tuple, list or raw numpy/cupy array)
            dest (int): Target process specifier.
            tag (int): Message ID (MPI feature).

        """
        chainer.utils.experimental(
            'chainermn.communicators.MpiCommunicatorBase.send')

        msgtype = _MessageType(data)
        _check_dtype('send', msgtype)

        """We use ssend() instead of send() to pass unittests.
        If we don't use it, an error occurs in
        test_point_to_point_communication.py
        when using MVAPICH2-2.2 and GPUs.
        """
        self.mpi_comm.ssend(msgtype, dest=dest, tag=tag)

        # Type check.
        if not msgtype.is_tuple:
            data = [data]

        for array in data:
            if numpy.float16 == array.dtype:
                array = array.astype(numpy.float32)

            if chainer.backend.get_array_module(array) is not numpy:
                chainer.cuda.Stream.null.synchronize()
                array = (_memory_utility.get_device_memory_pointer(array),
                         _get_mpi_type(msgtype))

            else:
                array = numpy.ascontiguousarray(array)

            """We use Ssend() for the same reason as using ssend()."""
            self.mpi_comm.Ssend(array, dest=dest, tag=tag)

    def recv(self, source, tag):
        """A primitive of inter-process receiver.

        This method tries to receive numpy-array from target process.
        The target process is expected to invoke ``send()``.
        This method relies on mpi4py fast communication optimized for
        numpy arrays, which discards any information attached to
        chainer.Variable objects. Please be sure.

        If the corresponding ``send()`` is invoked with cupy array,
        the returned array will be placed at current device
        (``https://docs-cupy.chainer.org/en/stable/tutorial/basic.html#current-device``)
        regardless of which device the argument is placed at remote nodes.

        Args:
            source (int): Target process specifier.
            tag (int): Message ID (MPI feature).

        Returns:
            data (tuple of numpy/cupy array or numpy/cupy array):
                Received data. If ``send()`` is invoked with tuple data,
                it is also tuple. Otherwise, it is a vanilla numpy/cupy array.
        """

        chainer.utils.experimental(
            'chainermn.communicators.MpiCommunicatorBase.recv')

        msgtype = self.mpi_comm.recv(source=source, tag=tag)
        xp = msgtype.get_array_module()
        if numpy.float16 == msgtype.dtype:
            comm_dtype = numpy.float32
        else:
            comm_dtype = msgtype.dtype

        if msgtype.is_tuple:
            msg = []
            for shape in msgtype.shapes:
                buf = xp.empty(
                    [chainer.utils.size_of_shape(shape)], dtype=comm_dtype)
                rtype = _get_mpi_type(msgtype)
                self.mpi_comm.Recv(
                    _memory_utility.array_to_buffer_object(buf, rtype),
                    source=source, tag=tag)

                if numpy.float16 == msgtype.dtype:
                    buf = buf.astype(numpy.float16)
                msg.append(buf.reshape(shape))
            return tuple(msg)

        else:
            assert len(msgtype.shapes) == 1
            shape = msgtype.shapes[0]
            buf = xp.empty([chainer.utils.size_of_shape(shape)],
                           dtype=comm_dtype)
            rtype = _get_mpi_type(msgtype)
            self.mpi_comm.Recv(
                _memory_utility.array_to_buffer_object(buf, rtype),
                source=source, tag=tag)

            if numpy.float16 == msgtype.dtype:
                buf = buf.astype(numpy.float16)
            return buf.reshape(shape)

    def bcast(self, x, root=0):
        """A primitive of inter-process broadcast communication.

        This method tries to invoke broadcast communication within the
        communicator. All processes in the communicator are expected to
        invoke ``broadcast()``. This method relies on mpi4py fast communication
        optimized for numpy arrays, as well as ``send()`` and ``recv()``.

        If ``bcast()`` is invoked with cupy array in the root process,
        the returned array will be placed at current device
        (``https://docs-cupy.chainer.org/en/stable/tutorial/basic.html#current-device``)
        regardless of which device the argument is placed at remote nodes.

        Args:
            x (numpy/cupy array): Array to be broadcasted.
            root (int): Rank of root process.

        Returns:
            ys (tuple of numpy/cupy array): Received arrays.
        """
        chainer.utils.experimental(
            'chainermn.communicators.MpiCommunicatorBase.bcast')

        is_master = self.mpi_comm.rank == root

        if is_master:
            msgtype = _MessageType(x)
            _check_dtype('bcast', msgtype)

            if msgtype.is_tuple:
                raise TypeError('Tuple data cannot be broadcasted')

            msgtype = self.mpi_comm.bcast(msgtype, root)
            shape = msgtype.shapes[0]
            buf = _memory_utility.array_to_buffer_object(
                x, _get_mpi_type(msgtype))
            self.mpi_comm.Bcast(buf, root)
            return x
        else:
            msgtype = self.mpi_comm.bcast(None, root)
            xp = msgtype.get_array_module()
            shape = msgtype.shapes[0]
            buf = xp.empty(
                [chainer.utils.size_of_shape(shape)], dtype=msgtype.dtype)
            buftype = _get_mpi_type(msgtype)
            self.mpi_comm.Bcast(
                _memory_utility.array_to_buffer_object(buf, buftype),
                root)
            return buf.reshape(shape)

    def gather(self, x, root=0):
        """A primitive of inter-process gather communication.

        This method tries to invoke gather communication within the
        communicator. All processes in the communicator are expected to
        invoke ``gather()``. This method relies on mpi4py fast communication
        optimized for numpy arrays, as well as ``send()`` and ``recv()``.

        If ``x`` is numpy array, the received data will also be allocated
        as numpy array. Additionally, when ``x`` is cupy array, the returned
        array will be placed at current device
        (``https://docs-cupy.chainer.org/en/stable/tutorial/basic.html#current-device``)
        regardless of which device the argument is placed at remote nodes.

        Args:
            x (numpy/cupy array): Array to be gathered.
            root (int): Rank of root process.

        Returns:
            ys (tuple of numpy/cupy array):
                Received arrays. ``None`` for non-root processes.
        """
        chainer.utils.experimental(
            'chainermn.communicators.MpiCommunicatorBase.gather')

        is_master = self.mpi_comm.rank == root

        msgtype = _MessageType(x)
        _check_dtype('gather', msgtype)

        msgtypes = self.mpi_comm.gather(msgtype, root)

        if is_master:
            _check_dtypes_are_same(msgtypes)

            for msgtype in msgtypes:
                if msgtype.is_tuple:
                    raise TypeError('gather cannot handle tuple data')

                assert len(msgtype.shapes) == 1

            xp = chainer.backend.get_array_module(x)
            sbuf = _memory_utility.array_to_buffer_object(
                x, _get_mpi_type(msgtype))
            shapes = [mty.shapes[0] for mty in msgtypes]
            rlens = [chainer.utils.size_of_shape(s) for s in shapes]
            rbuf = xp.empty([sum(rlens)], dtype=msgtype.dtype)

            if xp is not numpy:
                chainer.cuda.Stream.null.synchronize()

            self.mpi_comm.Gatherv(
                sbuf,
                [_memory_utility.get_device_memory_pointer(rbuf),
                 (rlens, _cnt_to_dsp(rlens)), _get_mpi_type(msgtype)],
                root)

            ys = [rbuf[i:i + l].reshape(s)
                  for i, l, s in zip(_cnt_to_dsp(rlens), rlens, shapes)]
            return tuple(ys)

        else:
            sbuf = _memory_utility.array_to_buffer_object(
                x, _get_mpi_type(msgtype))
            self.mpi_comm.Gatherv(sbuf, None, root)
            return None

    def allgather(self, x):
        chainer.utils.experimental(
            'chainermn.communicators.MpiCommunicatorBase.allgather')

        msgtype = _MessageType(x)
        _check_dtype('allgather', msgtype)

        msgtypes = self.mpi_comm.allgather(msgtype)
        _check_dtypes_are_same(msgtypes)

        # Type check.
        for msgtype in msgtypes:
            if msgtype.is_tuple:
                raise TypeError('allgather cannot handle tuple data')

            assert len(msgtype.shapes) == 1

        # Collective communication.
        xp = chainer.backend.get_array_module(x)
        shapes = [msgtype.shapes[0] for msgtype in msgtypes]
        sbuf = _memory_utility.array_to_buffer_object(
            x, _get_mpi_type(msgtype))
        rlens = [chainer.utils.size_of_shape(s) for s in shapes]
        rbuf = xp.empty([sum(rlens)], dtype=msgtype.dtype)
        if xp is not numpy:
            chainer.cuda.Stream.null.synchronize()
        self.mpi_comm.Allgatherv(
            sbuf,
            [_memory_utility.get_device_memory_pointer(rbuf),
             (rlens, _cnt_to_dsp(rlens)), _get_mpi_type(msgtype)])
        ys = [rbuf[i:i + l].reshape(s)
              for i, l, s in zip(_cnt_to_dsp(rlens), rlens, shapes)]

        return tuple(ys)

    def allreduce(self, x):
        """A primitive of inter-process allreduce communication.

        This method tries to invoke allreduce communication within the
        communicator. All processes in the communicator are expected to
        invoke ``allreduce()``. This method relies on mpi4py fast communication
        optimized for numpy arrays, as well as ``send()`` and ``recv()``.

        Note that this method can only handle the same shapes of data
        over all processes, and cannot handle tuple data.

        If ``x`` is numpy array, the received data will also be allocated
        as numpy array. Additionally, when ``x`` is cupy array, the returned
        array will be placed at current device
        (``https://docs-cupy.chainer.org/en/stable/tutorial/basic.html#current-device``)
        regardless of which device the argument is placed at remote nodes.

        Args:
            x (numpy/cupy array): An array to apply allreduce operation.

        Returns:
            ys (numpy/cupy array): An array that allreduce (currently SUM only)
                has been applied.

        """

        msgtype = _MessageType(x)
        _check_dtype('allreduce', msgtype)

        if msgtype.is_tuple:
            raise TypeError('allreduce cannot handle tuple data')

        xp = chainer.backend.get_array_module(x)

        # TODO(kuenishi): do we check all messages have same shape and dims?

        # Source buffer
        sbuf = _memory_utility.array_to_buffer_object(
            x, _get_mpi_type(msgtype))
        # Destination buffer and its object
        shape = msgtype.shapes[0]
        dbuf = xp.empty(
            [chainer.utils.size_of_shape(shape)], dtype=msgtype.dtype)
        dbuf_buffer_obj = _memory_utility.array_to_buffer_object(
            dbuf, _get_mpi_type(msgtype))
        self.mpi_comm.Allreduce(sbuf, dbuf_buffer_obj)

        return dbuf.reshape(shape)

    def scatter(self, xs, root=0):
        """A primitive of inter-process scatter communication.

        This method tries to invoke scatter communication within the
        communicator. All processes in the communicator are expected to
        invoke ``scatter()``. This method relies on mpi4py fast communication
        optimized for numpy arrays, as well as ``send()`` and ``recv()``.

        If ``xs`` is tuple, each element is send to different processes.
        The length of the tuple must be the same as the communicator size.
        If ``xs`` is ``numpy.ndarrray``, it is splitted with the first
        axis and sent to different processes. For slave processes, ``xs``
        is allowed to be any value (will be ignored).

        If ``scatter()`` is invoked with cupy array in the root process,
        the returned array will be placed at current device
        (``https://docs-cupy.chainer.org/en/stable/tutorial/basic.html#current-device``)
        regardless of which device the argument is placed at remote nodes.

        Args:
            xs (tuple of numpy/cupy array): Arrays to be scattered.
            root (int): Rank of root process.

        Returns:
            ys (numpy/cupy array): Received arrays.
        """
        chainer.utils.experimental(
            'chainermn.communicators.CommunicatorBase.scatter')

        is_master = self.mpi_comm.rank == root

        if is_master:
            # Type check.
            msgtype = _MessageType(xs)
            _check_dtype('scatter', msgtype)

            if msgtype.is_tuple:
                if len(msgtype.shapes) != self.size:
                    raise ValueError(
                        'the length of xs must be consistent '
                        'with communicator size')

                xp = chainer.backend.get_array_module(*xs)
                msgtype = tuple([_MessageType(x) for x in xs])
                shapes = [mty.shapes[0] for mty in msgtype]
                # concatenate([x.reshape(-1) ... ], axis=0) will fail
                xs = xp.concatenate([x.reshape(1, -1) for x in xs], axis=1)

            else:
                assert len(msgtype.shapes) == 1

                if msgtype.shapes[0][0] != self.mpi_comm.size:
                    raise ValueError(
                        'scatter received inconsistent number of inputs '
                        'with communicator size')

                xp = chainer.backend.get_array_module(xs)
                msgtype = tuple([_MessageType(xs[0])
                                 for _ in range(self.size)])
                shapes = [xs.shape[1:] for _ in range(self.size)]

            msgtype = self.mpi_comm.scatter(msgtype, root)
            shape = msgtype.shapes[0]

            # Collective communication.
            slens = [chainer.utils.size_of_shape(s) for s in shapes]
            sbuf = _memory_utility.get_device_memory_pointer(xs)
            rbuf = xp.empty(
                [chainer.utils.size_of_shape(shape)], dtype=msgtype.dtype)
            rtype = _get_mpi_type(msgtype)
            if xp is not numpy:
                chainer.cuda.Stream.null.synchronize()

            self.mpi_comm.Scatterv(
                [sbuf, (slens, _cnt_to_dsp(slens)), _get_mpi_type(msgtype)],
                _memory_utility.array_to_buffer_object(rbuf, rtype), root)

            return rbuf.reshape(shape)

        else:  # slave processes
            msgtypes = self.mpi_comm.scatter(None, root)
            xp = msgtypes.get_array_module()
            shape = msgtypes.shapes[0]
            rbuf = xp.empty(
                [chainer.utils.size_of_shape(shape)], dtype=msgtypes.dtype)
            rtype = _get_mpi_type(msgtypes)
            self.mpi_comm.Scatterv(
                None,
                _memory_utility.array_to_buffer_object(rbuf, rtype),
                root)
            return rbuf.reshape(shape)

    def _check_obj_type_for_chainerx(self, obj):
        # Do NOT support chainerx ndarray with CUDA
        # for the following reason:
        # (1) mpi4py.send pickles the object
        # (2) chainerx.ndarray preserves CUDA
        # device internally when pickled
        # (3) An error will occur when an ndarray is unpickled in another
        #     process
        #
        if None is obj:
            return False

        # check collections of list, tuple and set
        elif type(obj) in [list, tuple, set]:
            for item in obj:
                xp = chainer.backend.get_array_module(item)
                # DO NOT use device.backend.name as
                # 'ChainerxDevice' object has no attribute 'backend'
                if xp == chainerx and item.device.name.startswith('cuda'):
                    return True

        # check dict
        elif type(obj) is dict:
            for key, value in obj.items():
                xp = chainer.backend.get_array_module(key)
                if xp == chainerx and key.device.name.startswith('cuda'):
                    return True

                xp = chainer.backend.get_array_module(value)
                if xp == chainerx and value.device.name.startswith('cuda'):
                    return True
        else:
            xp = chainer.backend.get_array_module(obj)
            if xp == chainerx and obj.device.name.startswith('cuda'):
                return True

        return False

    # Objects
    def send_obj(self, obj, dest, tag=0):
        if self._check_obj_type_for_chainerx(obj):
            raise ValueError(
                'calling send_obj on chainerx \
                with cuda is not supported')
        self.mpi_comm.send(obj, dest=dest, tag=tag)

    def recv_obj(self, source, status=None, tag=mpi4py.MPI.ANY_TAG):
        return self.mpi_comm.recv(source=source, status=status, tag=tag)

    def bcast_obj(self, obj, max_buf_len=256 * 1024 * 1024, root=0):
        if self._check_obj_type_for_chainerx(obj):
            raise ValueError(
                'calling bcast_obj on chainerx \
                with cuda is not supported')
        return chunked_bcast_obj(obj, self.mpi_comm,
                                 max_buf_len=max_buf_len,
                                 root=root)

    def gather_obj(self, obj, root=0):
        if self._check_obj_type_for_chainerx(obj):
            raise ValueError(
                'calling gather_obj on chainerx \
                with cuda is not supported')
        return self.mpi_comm.gather(obj, root=root)

    def allreduce_obj(self, obj):
        # Summation by default
        if self._check_obj_type_for_chainerx(obj):
            raise ValueError(
                'calling allreduce_obj on chainerx \
                with cuda is not supported')
        return self.mpi_comm.allreduce(obj)

    def bcast_data(self, model):
        for _, param in sorted(model.namedparams()):
            if param.data is not None:
                data = param.data
                is_float16 = param.data.dtype == numpy.float16
                if is_float16:
                    data = data.astype(numpy.float32)
                buf = _memory_utility.array_to_buffer_object(data)
                self.mpi_comm.Bcast(buf)
                if is_float16:
                    # update to array as updating to .data directly
                    # is not supported in ChainerX
                    param.array[...] = data.astype(numpy.float16)

    # Private methods
    def _init_ranks(self):
        my_ranks = _communication_utility.init_ranks(self.mpi_comm)
        assert my_ranks[0] == self.mpi_comm.rank
        self._intra_rank = my_ranks[1]
        self._intra_size = my_ranks[2]
        self._inter_rank = my_ranks[3]
        self._inter_size = my_ranks[4]

    def _check_ready_to_allreduce(self, array_a, array_b):
        my_shapes = ((None if array_a is None else array_a.shape,
                      None if array_a is None else array_a.dtype),
                     array_b.shape,
                     array_b.dtype)
        all_shapes = self.gather_obj((self.rank, my_shapes))
        if self.rank == 0:
            for rank, shapes in all_shapes:
                if my_shapes != shapes:
                    raise ValueError('Shape does not match: {}'
                                     ' at rank 0 while {} at rank {}'
                                     .format(my_shapes, shapes, rank))

    def _ensure_all_finite(self, array):
        xp = chainer.backend.get_array_module(array)
        if not xp.isfinite(array).all():
            raise ValueError('Parameters diverged after allreduce.')

    def _multi_node_mean(self, sendbuf, recvbuf):
        """Compute mean of each element on each processes.

        The function compute mean of each element in ``sendbuf`` on each
        processes. The result is stored in ``recvbuf``.

        If ``sendbuf`` is ``None``, the function compute mean of each element
        in ``recvbuf`` on each processes and replaces ``recvbuf` with the
        computed mean.

        Args:
            sendbuf (numpy/cupy array): Input arrays.
            recvbuf (numpy/cupy array): Output arrays.

        """

        if chainer.is_debug():
            self._check_ready_to_allreduce(sendbuf, recvbuf)

        is_float16 = recvbuf.dtype == numpy.float16
        if sendbuf is None:
            buffer_a = mpi4py.MPI.IN_PLACE
        elif is_float16:
            assert sendbuf.dtype == recvbuf.dtype
            buffer_a = _memory_utility.array_to_buffer_object(
                sendbuf.astype(numpy.float32))
        else:
            buffer_a = _memory_utility.array_to_buffer_object(sendbuf)

        if is_float16:
            array_b32 = recvbuf.astype(numpy.float32)
        else:
            array_b32 = recvbuf
        buffer_b = _memory_utility.array_to_buffer_object(array_b32)

        self.mpi_comm.Allreduce(buffer_a, buffer_b)

        if is_float16:
            recvbuf[...] = array_b32.astype(numpy.float16)

        recvbuf *= 1.0 / self.mpi_comm.size

        if chainer.is_debug():
            self._ensure_all_finite(recvbuf)

    def _pack_params_to_buffer(self, params, attr_name, buffer,
                               allreduce_grad_dtype, zero_fill, stream=None):

        if self.batched_copy:
            params_data = _memory_utility.ParamsData(params,
                                                     attr_name, zero_fill)
            _memory_utility._batched_pack_params(
                params_data, buffer,
                allreduce_grad_dtype, stream=stream)
            self.params_data = params_data
        else:
            _memory_utility.pack_params(
                params, attr_name,
                buffer,
                transfer_dtype=allreduce_grad_dtype,
                zero_fill=zero_fill,
                stream=stream)

    def _unpack_params_from_buffer(self, params, attr_name, buffer,
                                   allreduce_grad_dtype,
                                   zero_fill, stream=None):
        if self.batched_copy:
            if self.params_data is not None:
                params_data = self.params_data
                self.params_data = None
            else:
                params_data = _memory_utility.ParamsData(
                    params, attr_name, zero_fill)
            _memory_utility._batched_unpack_params(
                params_data, buffer,
                allreduce_grad_dtype, stream=stream)
            return
        else:
            _memory_utility.unpack_params(
                params, attr_name, buffer,
                allreduce_grad_dtype, zero_fill, stream)
