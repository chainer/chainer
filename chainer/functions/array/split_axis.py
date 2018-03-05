import collections

import six

import chainer
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import function_node
from chainer.utils import type_check


class SplitAxis(function_node.FunctionNode):

    """Function that splits multiple arrays along the specified axis."""

    def __init__(self, indices_or_sections, axis):
        if not isinstance(
                indices_or_sections,
                six.integer_types + (collections.Iterable,)):
            raise TypeError('indices_or_sections must be integer or 1-D array')
        if (chainer.is_debug() and
                isinstance(indices_or_sections, collections.Iterable)):
            for p, n in six.moves.zip(
                    indices_or_sections, indices_or_sections[1:]):
                if p > n:
                    raise ValueError('indices_or_sections must be sorted')
        self.indices_or_sections = indices_or_sections
        self.axis = axis

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].ndim > self.axis)

        if isinstance(self.indices_or_sections, collections.Iterable):
            if len(self.indices_or_sections) > 0:
                max_index = type_check.make_variable(
                    self.indices_or_sections[-1], 'max_index')
                type_check.expect(in_types[0].shape[self.axis] > max_index)
        else:
            sections = type_check.make_variable(
                self.indices_or_sections, 'sections')
            type_check.expect(in_types[0].shape[self.axis] % sections == 0)

    def forward(self, inputs):
        # Currently iDeep only supports 4 dims
        if (intel64.should_use_ideep('>=auto')
                and intel64.inputs_all_ready(inputs, (4,))
                and self._ideep_is_supported(inputs)):
            return self._forward_ideep(inputs)

        x, = inputs
        if isinstance(self.indices_or_sections, collections.Iterable):
            cdimx = x.shape[self.axis]
            ind = list(self.indices_or_sections)
            ind.append(cdimx)
        self._xp = cuda.get_array_module(x)
        ret = tuple(self._xp.split(x, self.indices_or_sections, self.axis))
        self._shapes = [r.shape for r in ret]
        return ret

    def _ideep_is_supported(self, inputs):
        # Returns True if iDeep supports current configuration of inputs and
        # arguments. This is workaround for limitation in iDeep internal
        # implementation.
        ios = self.indices_or_sections
        if isinstance(ios, collections.Iterable):
            if len(ios) == 0:
                return False  # Empty sequence
            if ios[0] == 0:
                return False  # Sequence starting with 0
            for i in six.moves.range(1, len(ios)):
                if ios[i-1] == ios[i]:
                    return False  # Sequence with duplicate index
        else:
            if ios == 1:
                return False  # 1

        # Workaround for iDeep segfault issue
        # See:
        #   https://github.com/chainer/chainer/pull/4281#issuecomment-365830630
        # TODO(niboshi): Remove this after iDeep is fixed.
        # Note: inputs[0].ndim is always 4.
        if (self.axis == 1 or self.axis == -3) and inputs[0].shape[1] == 8:
            return False

        return True

    def _forward_ideep(self, inputs):
        x, = inputs
        offsets = intel64.ideep.intVector()
        # TODO(iDeep)
        # bypass python3 issue when transfer array to std::vector<>
        # https://github.com/SimpleITK/SimpleITK/issues/106
        ios = self.indices_or_sections
        axis = self.axis % x.ndim
        if isinstance(ios, collections.Iterable):
            for i in ios:
                offsets.push_back(int(i))
        else:
            d = x.shape[self.axis]
            step = d // ios
            for i in six.moves.range(step, d, step):
                offsets.push_back(i)
        ret = intel64.ideep.concat.Backward(
            intel64.ideep.array(x), offsets, axis)
        self._shapes = [r.shape for r in ret]
        return ret

    def backward(self, indexes, grad_outputs):
        dtype = self.inputs[0].dtype
        grads = [
            self._xp.zeros(shape, dtype=dtype) if gy is None else gy
            for gy, shape in six.moves.zip(grad_outputs, self._shapes)]
        return chainer.functions.concat(grads, self.axis),


def split_axis(x, indices_or_sections, axis, force_tuple=True):
    """Splits given variables along an axis.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            A variable to be split.
        indices_or_sections (int or 1-D array): If this argument is an integer,
            N, the array will be divided into N equal arrays along axis.
            If it is a 1-D array of sorted integers, it
            indicates the positions where the array is split.
        axis (int): Axis that the input array is split along.
        force_tuple (bool): If ``True`` (the default) this method returns a
            tuple even when the number of outputs is one. Otherwise, if
            ``False`` a Variable will be returned when the number of outputs
            is one.

    Returns:
        tuple or Variable: Tuple of :class:`~chainer.Variable` objects
        if the number of outputs is more than 1 or
        :class:`~chainer.Variable` otherwise.
        When ``force_tuple`` is ``True``, returned value is always a tuple
        regardless of the number of outputs.

    .. note::
        This function raises :class:`ValueError` if at least
        one of the outputs is split to zero-size
        (i.e. ``axis``-th value of its shape is zero).

    """
    res = SplitAxis(indices_or_sections, axis).apply((x,))
    if force_tuple or len(res) != 1:
        return res
    return res[0]
