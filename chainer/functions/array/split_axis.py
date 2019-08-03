import numpy
import six

import chainer
from chainer import backend
from chainer.backends import intel64
from chainer import function_node
from chainer.utils import collections_abc
from chainer.utils import type_check
import chainerx


_numpy_split_ok = numpy.lib.NumpyVersion(numpy.__version__) >= '1.11.0'


def _fix_numpy_split(ys, x, indices_or_sections, axis):
    """Make the output of np.split compatible with numpy >= 1.11"""
    if all(y.ndim == x.ndim for y in ys):
        return ys
    tmp = [len(t) for t in numpy.split(
        numpy.empty(x.shape[axis], dtype=numpy.int8), indices_or_sections, 0)]
    shape = list(x.shape)
    for i, t in enumerate(tmp):
        y = ys[i]
        if y.ndim != x.ndim:
            assert y.size == 0
            shape[axis] = t
            ys[i] = y.reshape(shape)
    return ys


def _get_indices_or_sections(indices_or_sections):
    """Checks and convert ``indices_or_sections`` argument

    Converted value is one of: 1-D numpy.ndarray, list, int, and
    NumPy int scalar.

    Returns:
        A binary tuple in which the 1st element is indices (sequence) and
        the 2nd element is sections (scalar).
        Only one of the two is not ``None`` and the other is ``None``.

    """
    ios = indices_or_sections
    is_seq = False
    if isinstance(ios, numpy.ndarray):
        # numpy.ndarray
        if ios.dtype.kind != 'i' and ios.size > 0:
            # Note: numpy.array([]) (dtype is float64) should be accepted.
            raise TypeError('indices_or_sections must be integers')
        if ios.ndim >= 2:
            raise TypeError('indices_or_sections must be 1-D sequence')
        is_seq = ios.ndim != 0
    elif isinstance(ios, collections_abc.Sequence):
        # Any sequence except numpy.ndarray
        ios = list(ios)
        is_seq = True
    elif isinstance(indices_or_sections, six.integer_types):
        # int
        pass
    else:
        raise TypeError(
            'indices_or_sections must be integer or 1-D array.\n'
            'Actual: {}'.format(type(indices_or_sections)))

    if is_seq and chainer.is_debug():
        for p, n in six.moves.zip(ios, ios[1:]):
            if p > n:
                raise ValueError('indices_or_sections must be sorted')

    if is_seq:
        return ios, None
    else:
        return None, ios


class SplitAxis(function_node.FunctionNode):

    """Function that splits multiple arrays along the specified axis."""

    def __init__(self, indices_or_sections, axis):
        indices, sections = _get_indices_or_sections(indices_or_sections)
        assert (indices is None) != (sections is None)
        self.indices = indices
        self.sections = sections
        self.axis = axis

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].ndim > self.axis)

        if self.indices is not None:
            indices = self.indices
            if len(indices) > 0:
                max_index = type_check.make_variable(indices[-1], 'max_index')
                type_check.expect(in_types[0].shape[self.axis] >= max_index)
        else:
            assert self.sections is not None
            sections = type_check.make_variable(self.sections, 'sections')
            type_check.expect(in_types[0].shape[self.axis] % sections == 0)

    @property
    def indices_or_sections(self):
        return self.indices if self.indices is not None else self.sections

    def forward_chainerx(self, inputs):
        x, = inputs
        return tuple(chainerx.split(x, self.indices_or_sections, self.axis))

    def forward(self, inputs):
        x, = inputs
        self._xp = backend.get_array_module(x)

        # Currently iDeep only supports 4 dims
        if (intel64.should_use_ideep('>=auto')
                and intel64.inputs_all_ready(inputs, (4,))
                and self._ideep_is_supported(inputs)):
            return self._forward_ideep(inputs)

        indices_or_sections = self.indices_or_sections
        ret = self._xp.split(x, indices_or_sections, self.axis)
        if self._xp == numpy and not _numpy_split_ok:
            ret = _fix_numpy_split(ret, x, indices_or_sections, self.axis)
        self._shapes = [r.shape for r in ret]
        return tuple(ret)

    def _ideep_is_supported(self, inputs):
        # Returns True if iDeep supports current configuration of inputs and
        # arguments. This is workaround for limitation in iDeep internal
        # implementation.
        if self.indices is not None:
            indices = self.indices
            if len(indices) == 0:
                return False  # Empty sequence
            if indices[0] == 0:
                return False  # Sequence starting with 0
            for i in six.moves.range(1, len(indices)):
                if indices[i-1] == indices[i]:
                    return False  # Sequence with duplicate index
        else:
            if self.sections == 1:
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
        axis = self.axis % x.ndim
        if self.indices is not None:
            for i in self.indices:
                offsets.push_back(int(i))
        else:
            d = x.shape[self.axis]
            step = d // self.sections
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
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
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
        tuple or ~chainer.Variable: Tuple of :class:`~chainer.Variable` objects
        if the number of outputs is more than 1 or
        :class:`~chainer.Variable` otherwise.
        When ``force_tuple`` is ``True``, returned value is always a tuple
        regardless of the number of outputs.

    """
    res = SplitAxis(indices_or_sections, axis).apply((x,))
    if force_tuple or len(res) != 1:
        return res
    return res[0]
