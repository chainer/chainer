import numpy
import six

from chainer import cuda

try:
    import cupy
    from cupy.cuda import runtime
except Exception:
    pass


def to_device(device, x):
    """Send an array to a given device.

    This method sends a given array to a given device. This method is used in
    :func:`~chainer.dataset.concat_examples`.
    You can also use this method in a custom converter method used in
    :class:`~chainer.training.Updater` and :class:`~chainer.training.Extension`
    such as :class:`~chainer.training.StandardUpdater` and
    :class:`~chainer.training.extensions.Evaluator`.

    See also :func:`chainer.dataset.concat_examples`.

    Args:
        device (int or None): Device ID to which an array is sent. If it is
            negative value, an array is sent to CPU. If it is positive, an
            array is sent to GPU with the given ID. If it is ``None``, an
            array is left in the original device.
        x (numpy.ndarray or cupy.ndarray): An array to send.

    Returns:
        Converted array.

    """
    if device is None:
        return x
    elif device < 0:
        return cuda.to_cpu(x)
    else:
        return cuda.to_gpu(x, device)


def concat_examples(batch, device=None, padding=None):
    """Concatenates a list of examples into array(s).

    Dataset iterator yields a list of examples. If each example is an array,
    this function concatenates them along the newly-inserted first axis (called
    `batch dimension`) into one array. The basic behavior is same for examples
    consisting of multiple arrays, i.e., corresponding arrays of all examples
    are concatenated.

    For instance, consider each example consists of two arrays ``(x, y)``.
    Then, this function concatenates ``x`` 's into one array, and ``y`` 's
    into another array, and returns a tuple of these two arrays. Another
    example: consider each example is a dictionary of two entries whose keys
    are ``'x'`` and ``'y'``, respectively, and values are arrays. Then, this
    function concatenates ``x`` 's into one array, and ``y`` 's into another
    array, and returns a dictionary with two entries ``x`` and ``y`` whose
    values are the concatenated arrays.

    When the arrays to concatenate have different shapes, the behavior depends
    on the ``padding`` value. If ``padding`` is ``None`` (default), it raises
    an error. Otherwise, it builds an array of the minimum shape that the
    contents of all arrays can be substituted to. The padding value is then
    used to the extra elements of the resulting arrays.

    TODO(beam2d): Add an example.

    Args:
        batch (list): A list of examples. This is typically given by a dataset
            iterator.
        device (int): Device ID to which each array is sent. Negative value
            indicates the host memory (CPU). If it is omitted, all arrays are
            left in the original device.
        padding: Scalar value for extra elements. If this is None (default),
            an error is raised on shape mismatch. Otherwise, an array of
            minimum dimensionalities that can accommodate all arrays is
            created, and elements outside of the examples are padded by this
            value.

    Returns:
        Array, a tuple of arrays, or a dictionary of arrays. The type depends
        on the type of each example in the batch.

    """
    if len(batch) == 0:
        raise ValueError('batch is empty')

    first_elem = batch[0]

    if isinstance(first_elem, tuple):
        result = []
        if not isinstance(padding, tuple):
            padding = [padding] * len(first_elem)

        for i in six.moves.range(len(first_elem)):
            result.append(to_device(device, _concat_arrays(
                [example[i] for example in batch], padding[i])))

        return tuple(result)

    elif isinstance(first_elem, dict):
        result = {}
        if not isinstance(padding, dict):
            padding = {key: padding for key in first_elem}

        for key in first_elem:
            result[key] = to_device(device, _concat_arrays(
                [example[key] for example in batch], padding[key]))

        return result

    else:
        return to_device(device, _concat_arrays(batch, padding))


def _concat_arrays(arrays, padding):
    # Convert `arrays` to numpy.ndarray if `arrays` consists of the built-in
    # types such as int or float.
    if not isinstance(arrays[0], numpy.ndarray) and\
       not isinstance(arrays[0], cuda.ndarray):
        arrays = numpy.asarray(arrays)
    if padding is not None:
        return _concat_arrays_with_padding(arrays, padding)

    xp = cuda.get_array_module(arrays[0])
    with cuda.get_device_from_array(arrays[0]):
        return xp.concatenate([array[None] for array in arrays])


def _concat_arrays_with_padding(arrays, padding):
    shape = numpy.array(arrays[0].shape, dtype=int)
    for array in arrays[1:]:
        if numpy.any(shape != array.shape):
            numpy.maximum(shape, array.shape, shape)
    shape = tuple(numpy.insert(shape, 0, len(arrays)))

    xp = cuda.get_array_module(arrays[0])
    with cuda.get_device_from_array(arrays[0]):
        result = xp.full(shape, padding, dtype=arrays[0].dtype)
        for i in six.moves.range(len(arrays)):
            src = arrays[i]
            slices = tuple(slice(dim) for dim in src.shape)
            result[(i,) + slices] = src

    return result


class ConcatWithAsyncTransfer(object):

    """Interface to concatenate data and transfer them to GPU asynchronously.

    It enables to transfer next batch of input data to GPU while GPU is
    running kernels for training using current batch of input data.

    Args:
        stream (cupy.cuda.Stream): CUDA stream. If ``None``, a stream is
            automatically created on the first call. Data transfer operation
            is launched acynchrnously using the stream.
    """

    def __init__(self, stream=None):
        self.stream = stream
        self._pin_arrays = [[], []]  # numpy arrays with pinned memory
        self._cp_arrays = [[], []]  # cupy arrays

    def __call__(self, batch, device=None, padding=None):
        """Concatenate data and transfer them to GPU asynchronously.

        (see concat_examples for detail about concatenation)

        Args:
            batch (list): A list of examples.
            device (int): Device ID to which each array is sent.
            padding: Scalar value for extra elements.

        Returns:
            Array, a tuple of arrays, or a dictionary of arrays.
            The type depends on the type of each example in the batch.
        """
        if len(batch) == 0:
            raise ValueError('batch is empty')

        first_elem = batch[0]
        if (device is None or
                not isinstance(first_elem, tuple)):
            return concat_examples(batch, device, padding)

        if not isinstance(padding, tuple):
            padding = [padding] * len(first_elem)

        pin_arrays = self._pin_arrays.pop(0)
        cp_arrays = self._cp_arrays.pop(0)

        with cuda.get_device_from_id(device):
            if self.stream is None:
                self.stream = cuda.Stream(non_blocking=True)
            stream = self.stream

            # Concatenate examples in batch into an array and check if
            # the size of the array matches the size of previously created
            # and retained array.
            _same_size = True
            np_arrays = []
            for i in six.moves.range(len(first_elem)):
                np_arrays.append(_concat_arrays(
                    [example[i] for example in batch], padding[i]))
                if (len(cp_arrays) <= i or
                        np_arrays[i].nbytes != cp_arrays[i].nbytes):
                    _same_size = False

            if not _same_size:
                pin_arrays = []
                cp_arrays = []

            if len(cp_arrays) == 0:
                # Allocate memory for numpy arrays with pinned memory
                # and cupy arrays. These arrays are retaind at
                # self._pin_arrays and self._cp_arrays respectively and
                # will be reused for subsequent batches as long as
                # the size are the same.

                # The global synchronization below is necceary to ensure ALL
                # operations including compute and data transfer submitted
                # to GPU so far have been completed, in order to avoid possible
                # memory corruption due to race condition among operations that
                # use different CUDA streams.
                # You can also solve this sort of race condition by preparing a
                # memory pool for each CUDA stream and using it carefully.
                runtime.deviceSynchronize()
                for i in six.moves.range(len(first_elem)):
                    np_array = np_arrays[i]
                    pin_mem = cupy.cuda.alloc_pinned_memory(np_array.nbytes)
                    pin_array = numpy.frombuffer(pin_mem,
                                                 np_array.dtype,
                                                 np_array.size
                                                 ).reshape(np_array.shape)
                    cp_array = cupy.empty_like(np_array)

                    pin_arrays.append(pin_array)
                    cp_arrays.append(cp_array)

            results = []
            for i in six.moves.range(len(first_elem)):
                pin_arrays[i][...] = np_arrays[i]  # copy(CPU): paged -> pinned
                cp_arrays[i].set(pin_arrays[i], stream)  # copy: CPU to GPU
                results.append(cp_arrays[i])

            # Wait for completion of the data transfer submitted above.
            # Global synchronizaton is used here for safer reason.
            # If a caller function is correctly handling the synchronization,
            # local synchronization (stream.synchronize()) may be enough.
            runtime.deviceSynchronize()

            self._pin_arrays.append(pin_arrays)
            self._cp_arrays.append(cp_arrays)

            return tuple(results)
