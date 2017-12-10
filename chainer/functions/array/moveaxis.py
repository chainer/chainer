import six

from chainer import cuda
from chainer import function_node
from chainer.utils import type_check


def _normalize_axis_tuple(axis, ndim, xp):
    if xp.isscalar(axis):
        axis = (axis,)

    ret = []
    for ax in axis:
        ret.append(ax % ndim)
    return ret


def _moveaxis(a, source, destination, xp):
    if hasattr(xp, 'moveaxis'):
        return xp.moveaxis(a, source, destination)
    source = _normalize_axis_tuple(source, a.ndim, xp)
    destination = _normalize_axis_tuple(destination, a.ndim, xp)
    order = [n for n in six.moves.range(a.ndim) if n not in source]

    for dest, src in sorted(six.moves.zip(destination, source)):
        order.insert(dest, src)

    result = a.transpose(order)
    return result


class Moveaxis(function_node.FunctionNode):

    """Move axis of an array."""

    def __init__(self, source, destination):
        if isinstance(source, int):
            if not isinstance(destination, int):
                raise ValueError('Types of source and destination are '
                                 'different.')
            self.source = (source,)
            self.destination = (destination,)
        elif isinstance(source, tuple) and all(isinstance(a, int)
                                               for a in source):
            if not isinstance(destination, tuple):
                raise ValueError('Types of source and destination are '
                                 'different.')
            if len(source) != len(destination):
                raise ValueError('Length of source and destination are '
                                 'different.')
            if len(set(source)) != len(source):
                raise ValueError('duplicate value in source axis: ({})'.format(
                    ', '.join(map(str, source))))
            if len(set(destination)) != len(destination):
                raise ValueError('duplicate value in destination axis: ({})'
                                 .format(', '.join(map(str, destination))))
            self.source = source
            self.destination = destination
        else:
            raise ValueError('int or tuple of int are required')

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
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
        xp = cuda.get_array_module(*inputs)
        return _moveaxis(inputs[0], self.source, self.destination, xp),

    def backward(self, indexes, gy):
        return Moveaxis(self.destination, self.source).apply(gy)


def moveaxis(x, source, destination):
    """Move the source axis to the destination.

    Args:
        x (~chainer.Variable): Input variable.
        source (int or tuple of int):
            Original positions of the axes to move. These must be unique.
        destination (int or tuple of int):
            Destination positions for each of the original axes.
            These must also be unique.

    Returns:
        ~chainer.Variable: Variable whose axis is moved.
    """
    return Moveaxis(source, destination).apply((x,))[0]
