import numpy
import six

from chainer import cuda


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
            minimum dimensionalities that can accomodate all arrays is created,
            and elements outside of the examples are padded by this value.

    Returns:
        Array, a tuple of arrays, or a dictionary of arrays. The type depends
        on the type of each example in the batch.

    """
    if len(batch) == 0:
        raise ValueError('batch is empty')

    if device is None:
        def to_device(x):
            return x
    elif device < 0:
        to_device = cuda.to_cpu
    else:
        to_device = lambda x: cuda.to_gpu(x, device)

    first_elem = batch[0]

    if isinstance(first_elem, tuple):
        result = []
        if not isinstance(padding, tuple):
            padding = [padding] * len(first_elem)

        for i in six.moves.range(len(first_elem)):
            result.append(to_device(_concat_arrays(
                [example[i] for example in batch], padding[i])))

        return tuple(result)

    elif isinstance(first_elem, dict):
        result = {}
        if not isinstance(padding, dict):
            padding = {key: padding for key in first_elem}

        for key in first_elem:
            result[key] = to_device(_concat_arrays(
                [example[key] for example in batch], padding[key]))

        return result

    else:
        return to_device(_concat_arrays(batch, padding))


def _concat_arrays(arrays, padding):
    if padding is not None:
        return _concat_arrays_with_padding(arrays, padding)

    xp = cuda.get_array_module(arrays[0])
    with cuda.get_device(arrays[0]):
        return xp.concatenate([array[None] for array in arrays])


def _concat_arrays_with_padding(arrays, padding):
    shape = numpy.array(arrays[0].shape, dtype=int)
    for array in arrays[1:]:
        if numpy.any(shape != array.shape):
            numpy.maximum(shape, array.shape, shape)
    shape = tuple(numpy.insert(shape, 0, len(arrays)))

    xp = cuda.get_array_module(arrays[0])
    with cuda.get_device(arrays[0]):
        result = xp.full(shape, padding, dtype=arrays[0].dtype)
        for i in six.moves.range(len(arrays)):
            src = arrays[i]
            slices = tuple(slice(dim) for dim in src.shape)
            result[(i,) + slices] = src

    return result
