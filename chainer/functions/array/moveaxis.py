import six

from chainer import backend
from chainer import function_node
from chainer.utils import type_check


def _normalize_axis_tuple(axis, ndim):
    ret = []
    for ax in axis:
        ret.append(ax % ndim)
    return ret


def _moveaxis(a, source, destination, xp):
    if hasattr(xp, 'moveaxis'):
        return xp.moveaxis(a, source, destination)
    if not all(isinstance(a, int) for a in source):
        raise TypeError('int or tuple of int are required.')
    if not all(isinstance(a, int) for a in destination):
        raise TypeError('int or tuple of int are required.')
    if len(source) != len(destination):
        raise ValueError('Length of source and destination are '
                         'different.')

    source = _normalize_axis_tuple(source, a.ndim)
    destination = _normalize_axis_tuple(destination, a.ndim)

    if len(set(source)) != len(source):
        raise ValueError('duplicate value in source axis: ({})'.format(
            ', '.join(map(str, source))))
    if len(set(destination)) != len(destination):
        raise ValueError('duplicate value in destination axis: ({})'
                         .format(', '.join(map(str, destination))))

    order = [n for n in six.moves.range(a.ndim) if n not in source]

    for dest, src in sorted(six.moves.zip(destination, source)):
        order.insert(dest, src)

    result = a.transpose(order)
    return result


class Moveaxis(function_node.FunctionNode):

    """Move axis of an array."""

    def __init__(self, source, destination):
        if isinstance(source, int):
            self.source = (source,)
        else:
            self.source = source

        if isinstance(destination, int):
            self.destination = (destination,)
        else:
            self.destination = destination

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(
            in_types[0].dtype.kind == 'f',
        )

        if self.source is not None:
            for axis in self.source:
                if axis >= 0:
                    type_check.expect(
                        axis < in_types[0].ndim,
                    )
                else:
                    type_check.expect(
                        -axis - 1 < in_types[0].ndim,
                    )
        if self.destination is not None:
            for axis in self.destination:
                if axis >= 0:
                    type_check.expect(
                        axis < in_types[0].ndim,
                    )
                else:
                    type_check.expect(
                        -axis - 1 < in_types[0].ndim,
                    )

    def forward(self, inputs):
        self.retain_inputs(())
        self._in_ndim = inputs[0].ndim
        xp = backend.get_array_module(*inputs)
        return _moveaxis(inputs[0], self.source, self.destination, xp),

    def backward(self, indexes, gy):
        return Moveaxis(self.destination, self.source).apply(gy)


def moveaxis(x, source, destination):
    """Move the source axes to the destination.

    This function transpose the input ``x`` by moving
    the axes ``source`` to the axes ``destination``.
    Other axes remain in their original order.

    See also :func:`chainer.functions.transpose`,
    :func:`chainer.functions.swapaxes`.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.
        source (int or tuple of int):
            Original positions of the axes to move. These must be unique.
        destination (int or tuple of int):
            Destination positions for each of the original axes.
            These must also be unique.

    Returns:
        ~chainer.Variable: Variable whose axis is moved.

    .. admonition:: Example

        >>> x = np.zeros((2, 3, 4, 5), np.float32)
        >>> chainer.functions.moveaxis(x, 0, -1).shape
        (3, 4, 5, 2)
        >>> chainer.functions.moveaxis(x, (0, 3), (2, 0)).shape
        (5, 3, 2, 4)

    """
    return Moveaxis(source, destination).apply((x,))[0]
