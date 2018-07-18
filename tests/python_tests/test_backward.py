import xchainer

import pytest


def _assert_arrays_equal(array1, array2):
    if array1 is None:
        assert array1 == array2
    else:
        assert array1.dtype == array2.dtype
        assert array1.shape == array2.shape
        assert array1._debug_flat_data == array2._debug_flat_data


def _check_backprop(xs, expected_gxs, fprop, extra_xs, graph_id=None):
    # Checks for test validity
    assert isinstance(xs, tuple)
    assert isinstance(expected_gxs, tuple)
    assert callable(fprop)
    assert isinstance(extra_xs, tuple)
    assert len(xs) == len(expected_gxs)
    assert all([isinstance(a, xchainer.ndarray) for a in xs])
    assert all([(isinstance(a, xchainer.ndarray) or a == xchainer.XchainerError) for a in expected_gxs])
    assert all([isinstance(a, xchainer.ndarray) for a in extra_xs])

    outputs = fprop(xs, extra_xs)
    xchainer.backward(outputs, graph_id)

    for i, expected_gx in enumerate(expected_gxs):
        x = xs[i]
        if expected_gx is xchainer.XchainerError:
            with pytest.raises(xchainer.XchainerError):
                x.get_grad(graph_id)
        else:
            gx = x.get_grad(graph_id)
            _assert_arrays_equal(gx, expected_gx)

    for output in outputs:
        grad = output.get_grad(graph_id)
        assert grad is not None


def test_backward_identity():
    shape = (1,)
    dtype = xchainer.float32

    xs = (xchainer.full(shape, 5, dtype),)
    expected_gxs = (xchainer.full(shape, 1, dtype),)

    for x in xs:
        x.require_grad()

    def fprop(xs_, extra_xs_):
        x, = xs_
        y = x.copy()
        return y,

    _check_backprop(xs, expected_gxs, fprop, ())


def test_backward_add():
    shape = (1,)
    dtype = xchainer.float32

    xs = (
        xchainer.full(shape, 3, dtype),
        xchainer.full(shape, 5, dtype),)
    expected_gxs = (
        xchainer.full(shape, 1, dtype),
        xchainer.full(shape, 1, dtype),)

    for x in xs:
        x.require_grad()

    def fprop(xs_, extra_xs_):
        x0, x1 = xs_
        y = x0 + x1
        return y,

    _check_backprop(xs, expected_gxs, fprop, ())


def test_backward_mul():
    shape = (1,)
    dtype = xchainer.float32

    xs = (
        xchainer.full(shape, 3, dtype),
        xchainer.full(shape, 5, dtype),)
    expected_gxs = (
        xchainer.full(shape, 5, dtype),
        xchainer.full(shape, 3, dtype),)

    for x in xs:
        x.require_grad()

    def fprop(xs_, extra_xs_):
        x0, x1 = xs_
        y = x0 * x1
        return y,

    _check_backprop(xs, expected_gxs, fprop, ())


def test_backward_add_mull():
    shape = (1,)
    dtype = xchainer.float32

    xs = (
        xchainer.full(shape, 2, dtype),
        xchainer.full(shape, 9, dtype),
        xchainer.full(shape, 5, dtype),)
    expected_gxs = (
        xchainer.full(shape, 14, dtype),
        xchainer.full(shape, 2, dtype),
        xchainer.full(shape, 2, dtype))

    for x in xs:
        x.require_grad()

    def fprop(xs_, extra_xs_):
        x0, x1, x2 = xs_
        y = x0 * (x1 + x2)
        return y,

    _check_backprop(xs, expected_gxs, fprop, ())


def test_backward_add_mul_extra_inputs():
    shape = (1,)
    dtype = xchainer.float32

    xs = (
        xchainer.full(shape, 2, dtype),
        xchainer.full(shape, 3, dtype))
    extra_xs = (xchainer.full(shape, 4, dtype),)
    expected_gxs = (
        xchainer.full(shape, 3, dtype),
        xchainer.full(shape, 6, dtype))

    for x in xs:
        x.require_grad()

    def fprop(xs_, extra_xs_):
        x0, x1 = xs_
        t0, = extra_xs_
        y = x1 * (x0 + t0)
        return y,

    _check_backprop(xs, expected_gxs, fprop, extra_xs)


def test_backward_sole_array_node():
    shape = (1,)
    dtype = xchainer.float32

    x = xchainer.full(shape, 2, dtype)
    expected_gx = xchainer.full(shape, 1, dtype)

    x.require_grad()

    xchainer.backward(x)

    _assert_arrays_equal(x.get_grad(), expected_gx)


def test_double_backprop():
    shape = (1,)
    dtype = xchainer.float32

    xs = (xchainer.full(shape, 2, dtype),)
    extra_xs = (xchainer.full(shape, 3, dtype),)
    expected_gxs = (xchainer.full(shape, 2, dtype),)

    for x in xs:
        x.require_grad()

    def fprop(xs_, extra_xs_):
        x, = xs_
        t, = extra_xs_
        y = x * (x + t)
        xchainer.backward(y, enable_double_backprop=True)
        gx = x.get_grad()  # 2x + y
        x.cleargrad()
        return gx,

    _check_backprop(xs, expected_gxs, fprop, extra_xs)


def test_multiple_graphs_double_backprop():
    with xchainer.graph_scope('graph_x') as graph_x, \
            xchainer.graph_scope('graph_y') as graph_y:

        x = xchainer.full((1,), 2, xchainer.float32)
        x.require_grad(graph_id=graph_x)

        y = xchainer.full((1,), 3, xchainer.float32)
        y.require_grad(graph_id=graph_y)

        z = x * (x + y)
        xchainer.backward(z, graph_id=graph_x)

        gx = x.get_grad(graph_x)  # 2x + y
        assert not gx.is_grad_required(graph_id=graph_x)
        assert gx.is_grad_required(graph_id=graph_y)

        w = x * gx
        xchainer.backward(w, graph_id=graph_y)

        e = xchainer.full((1,), 2, xchainer.float32)

        _assert_arrays_equal(y.get_grad(graph_y), e)  # x


def test_backward_input_to_multiple_ops():
    shape = (1,)
    dtype = xchainer.float32

    xs = (xchainer.full(shape, 2, dtype),)
    extra_xs = (xchainer.full(shape, 3, dtype),)
    expected_gxs = (xchainer.full(shape, 7, dtype),)

    for x in xs:
        x.require_grad()

    def fprop(xs_, extra_xs_):
        x, = xs_
        t, = extra_xs_
        y = x * (x + t)
        return y,

    _check_backprop(xs, expected_gxs, fprop, extra_xs)


def test_backward_identical_inputs():
    shape = (1,)
    dtype = xchainer.float32

    xs = (xchainer.full(shape, 2, dtype),)
    expected_gxs = (xchainer.full(shape, 2, dtype),)

    for x in xs:
        x.require_grad()

    def fprop(xs_, extra_xs_):
        x, = xs_
        y = x + x
        return y,

    _check_backprop(xs, expected_gxs, fprop, ())


def test_backward_identical_intermediate_nodes():
    shape = (1,)
    dtype = xchainer.float32

    xs = (xchainer.full(shape, 2, dtype),)
    expected_gxs = (xchainer.full(shape, 4, dtype),)

    for x in xs:
        x.require_grad()

    def fprop(xs_, extra_xs_):
        x, = xs_
        y = x + x
        z = y + y
        return z,

    _check_backprop(xs, expected_gxs, fprop, ())


def test_backward_given_input_grad():
    shape = (1,)
    dtype = xchainer.float32

    xs = (xchainer.full(shape, 1, dtype),)
    expected_gxs = (xchainer.full(shape, 2, dtype),)

    for x in xs:
        x.require_grad()

    def fprop(xs_, extra_xs_):
        x, = xs_
        x.set_grad(xchainer.full(shape, 1, dtype))
        y = x.copy()
        return y,

    _check_backprop(xs, expected_gxs, fprop, ())


def test_backward_given_output_grad():
    shape = (1,)
    dtype = xchainer.float32

    xs = (xchainer.full(shape, 2, dtype),)
    extra_xs = (xchainer.full(shape, 3, dtype),)
    expected_gxs = (xchainer.full(shape, 6, dtype),)

    for x in xs:
        x.require_grad()

    def fprop(xs_, extra_xs_):
        x, = xs_
        t, = extra_xs_
        y = x * t
        y.set_grad(xchainer.full(shape, 2, dtype))
        return y,

    _check_backprop(xs, expected_gxs, fprop, extra_xs)


def test_backward_keyword_arguments():
    x = xchainer.full((1,), 2, xchainer.float32)
    with xchainer.graph_scope('graph_1') as graph_id1:
        x.require_grad(graph_id=graph_id1)
        xchainer.backward(x, graph_id=graph_id1)
        with pytest.raises(TypeError, match=r'.*incompatible function arguments.*'):
            xchainer.backward(body=x, graph_id=graph_id1)


def test_backward_multiple_graphs_basic():
    shape = (1,)
    dtype = xchainer.float32

    x1 = xchainer.full(shape, 2, dtype)
    x2 = xchainer.full(shape, 5, dtype)

    with xchainer.graph_scope('graph_1') as graph_id1, \
            xchainer.graph_scope('graph_2') as graph_id2:

        x1.require_grad(graph_id1)
        x2.require_grad(graph_id2)

        xs = (x1, x2)
        expected_gxs = (xchainer.full(shape, 5, dtype), xchainer.XchainerError)

        def fprop(xs_, extra_xs_):
            x1, x2 = xs_
            y = x1 * x2
            return y,

        _check_backprop(xs, expected_gxs, fprop, (), graph_id1)


def test_backward_multiple_graphs_non_existing():
    shape = (1,)
    dtype = xchainer.float32

    x1 = xchainer.full(shape, 2, dtype)
    x2 = xchainer.full(shape, 5, dtype)

    with xchainer.graph_scope('graph_1') as graph_id1, \
            xchainer.graph_scope('graph_2') as graph_id2:

        x1.require_grad(graph_id1)
        x2.require_grad(graph_id1)

        y = x1 * x2
        with pytest.raises(xchainer.XchainerError):
            xchainer.backward(y, graph_id2)


def test_backward_multiple_graphs_reuse():
    shape = (1,)
    dtype = xchainer.float32

    x1 = xchainer.full(shape, 2, dtype)
    x2 = xchainer.full(shape, 5, dtype)

    with xchainer.graph_scope('graph_1') as graph_id1, \
            xchainer.graph_scope('graph_2') as graph_id2:

        x1.require_grad(graph_id1)
        x2.require_grad(graph_id2)

        xs = (x1, x2)

        def fprop(xs_, extra_xs_):
            x1, x2 = xs_
            y = x1 * x2
            return y,

        expected_gxs = (xchainer.full(shape, 5, dtype), xchainer.XchainerError)
        _check_backprop(xs, expected_gxs, fprop, (), graph_id1)

        x1.cleargrad(graph_id1)
        x2.cleargrad(graph_id2)

        assert x1.get_grad(graph_id1) is None
        assert x2.get_grad(graph_id2) is None

        expected_gxs = (xchainer.XchainerError, xchainer.full(shape, 2, dtype))
        _check_backprop(xs, expected_gxs, fprop, (), graph_id2)

        x1.cleargrad(graph_id1)
        x2.cleargrad(graph_id2)

        x1.require_grad(graph_id2)
        x2.require_grad(graph_id1)

        expected_gxs = (xchainer.full(shape, 5, dtype), xchainer.full(shape, 2, dtype))
        _check_backprop(xs, expected_gxs, fprop, (), graph_id2)

        assert x1.get_grad(graph_id1) is None
        assert x2.get_grad(graph_id1) is None


def test_backward_multiple_outputs():
    shape = (1,)
    dtype = xchainer.float32

    xs = (
        xchainer.full(shape, 3, dtype),
        xchainer.full(shape, 5, dtype),)
    expected_gxs = (
        xchainer.full(shape, 6, dtype),
        xchainer.full(shape, 4, dtype),)

    for x in xs:
        x.require_grad()

    def fprop(xs_, extra_xs_):
        x0, x1 = xs_
        return (x0 + x1, x0 * x1)

    _check_backprop(xs, expected_gxs, fprop, ())
