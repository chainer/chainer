import chainer
from chainer import function_node
from chainer.utils import type_check


# TODO(niboshi): memory layouts and conversions should better be implemented
# with polymorphism instead of ad-hoc conditions.

CUDNN_CHANNEL_FIRST_X = None
CUDNN_CHANNEL_LAST_X = 'CUDNN_CHANNEL_LAST_X'
CUDNN_CHANNEL_FIRST_W = None
CUDNN_CHANNEL_LAST_W = 'CUDNN_CHANNEL_LAST_W'


class _Unspecified:
    def __repr__(self):
        return '<unspecified>'


_unspecified = _Unspecified()


def get_raw_shape(arr_or_var):
    if isinstance(arr_or_var, chainer.Variable):
        arr = arr_or_var._data[0]
    else:
        arr = arr_or_var
    return arr.shape


def get_semantic_shape(arr_or_var, *, assumed_layout=_unspecified):
    if not isinstance(arr_or_var, chainer.Variable):
        # array
        shape = arr_or_var.shape
        if assumed_layout is not _unspecified:
            shape = _transpose_shape(shape, assumed_layout, None)
        return shape

    # variable
    if assumed_layout is not _unspecified:
        # TODO(niboshi): Raise exception
        assert arr_or_var.layout == assumed_layout
    return arr_or_var.shape


def _transpose_array(arr, src_layout, dst_layout):
    trans = _get_layout_transpose_axes(arr.ndim, src_layout, dst_layout)
    if trans is None:
        return arr
    return arr.transpose(*trans)


def _transpose_shape(shape, src_layout, dst_layout):
    trans = _get_layout_transpose_axes(len(shape), src_layout, dst_layout)
    if trans is None:
        return shape
    return tuple([shape[i] for i in trans])


def _get_layout_transpose_axes(ndim, src_layout, dst_layout, inverse=False):
    # None: no transposition is required.

    if src_layout == dst_layout:
        return None

    if dst_layout == CUDNN_CHANNEL_LAST_X:
        assert ndim >= 3
        assert src_layout == CUDNN_CHANNEL_FIRST_X
        trans = (0,) + tuple(range(2, ndim)) + (1,)

    elif dst_layout == CUDNN_CHANNEL_LAST_W:
        assert ndim >= 3
        assert src_layout == CUDNN_CHANNEL_FIRST_W
        trans = (0,) + tuple(range(2, ndim)) + (1,)

    elif src_layout == CUDNN_CHANNEL_LAST_X:
        assert ndim >= 3
        assert dst_layout == CUDNN_CHANNEL_FIRST_X
        trans = (0, ndim-1) + tuple(range(1, ndim-1))

    elif src_layout == CUDNN_CHANNEL_LAST_W:
        assert ndim >= 3
        assert dst_layout == CUDNN_CHANNEL_FIRST_W
        trans = (0, ndim-1) + tuple(range(1, ndim-1))

    else:
        raise ValueError(
            'Unknown layout conversion: from \'{}\' to \'{}\''.format(
                src_layout, dst_layout))

    if inverse:
        t = [None] * ndim
        for i, n in enumerate(trans):
            t[n] = i
        trans = tuple(t)

    # Postconditions:
    # - assert isinstance(trans, tuple)
    # - assert len(trans) == ndim
    # - assert all([i in trans for i in range(ndim)])
    return trans


class AsLayout(function_node.FunctionNode):
    """Permute the dimensions of an array."""

    axes = None
    in_layout = None

    def __init__(self, layout):
        self.out_layout = layout

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1,)

    def check_layout_forward(self, inputs):
        x, = inputs
        self.axes = _get_layout_transpose_axes(
            x.ndim, x.layout, self.out_layout)
        self.in_layout = x.layout

    @property
    def label(self):
        return 'AsLayout'

    def forward_chainerx(self, inputs):
        # TODO(niboshi): Add support for this
        raise RuntimeError(
            'Non-standard memory layouts are not supported for chainerx.')

    def forward(self, inputs):
        x, = inputs
        axes = self.axes
        self.output_layouts = (self.out_layout,)
        if axes is None:
            return x
        return x.transpose(axes),

    def backward(self, indexes, grad_outputs):
        return AsLayout(self.in_layout).apply(grad_outputs)
