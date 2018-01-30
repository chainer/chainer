import xchainer

import pytest


def assert_arrays_equal(array1, array2):
    if array1 is None:
        assert array1 == array2
    else:
        assert array1.dtype == array2.dtype
        assert array1.shape == array2.shape
        assert array1._debug_flat_data == array2._debug_flat_data


def check_backprop(xs, expected_gxs, fprop, extra_xs, graph_id=''):
    # Checks for test validity
    assert isinstance(xs, tuple)
    assert isinstance(expected_gxs, tuple)
    assert callable(fprop)
    assert isinstance(extra_xs, tuple)
    assert len(xs) == len(expected_gxs)
    assert all([isinstance(a, xchainer.Array) for a in xs])
    assert all([(isinstance(a, xchainer.Array) or a == xchainer.XchainerError) for a in expected_gxs])
    assert all([isinstance(a, xchainer.Array) for a in extra_xs])

    outputs = fprop(xs, extra_xs)
    assert len(outputs) == 1, 'This test does not support multi-output functions yet'

    xchainer.backward(outputs[0], graph_id)

    for i, expected_gx in enumerate(expected_gxs):
        x = xs[i]
        if expected_gx == xchainer.XchainerError:
            with pytest.raises(xchainer.XchainerError):
                x.get_grad(graph_id)
        else:
            gx = x.get_grad(graph_id)
            assert_arrays_equal(gx, expected_gx)

    assert outputs[0].get_grad(graph_id) is not None


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

    check_backprop(xs, expected_gxs, fprop, ())


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

    check_backprop(xs, expected_gxs, fprop, ())


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

    check_backprop(xs, expected_gxs, fprop, ())


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

    check_backprop(xs, expected_gxs, fprop, ())


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

    check_backprop(xs, expected_gxs, fprop, extra_xs)


def test_backward_sole_array_node():
    shape = (1,)
    dtype = xchainer.float32

    x = xchainer.full(shape, 2, dtype)
    expected_gx = xchainer.full(shape, 1, dtype)

    x.require_grad()

    xchainer.backward(x)

    assert_arrays_equal(x.get_grad(), expected_gx)


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
        xchainer.backward(y)
        gx = x.get_grad()
        x.set_grad(None)
        return gx,

    check_backprop(xs, expected_gxs, fprop, extra_xs)


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

    check_backprop(xs, expected_gxs, fprop, extra_xs)


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

    check_backprop(xs, expected_gxs, fprop, ())


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

    check_backprop(xs, expected_gxs, fprop, ())


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

    check_backprop(xs, expected_gxs, fprop, ())


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

    check_backprop(xs, expected_gxs, fprop, extra_xs)


def test_backward_keyword_arguments():
    x = xchainer.full((1,), 2, xchainer.float32)
    graph_id1 = 'graph_1'
    x.require_grad(graph_id=graph_id1)
    xchainer.backward(x, graph_id=graph_id1)
    with pytest.raises(TypeError, match=r'.*incompatible function arguments.*'):
        xchainer.backward(body=x, graph_id=graph_id1)


def test_backward_multiple_graphs_basic():
    shape = (1,)
    dtype = xchainer.float32

    x1 = xchainer.full(shape, 2, dtype)
    x2 = xchainer.full(shape, 5, dtype)

    graph_id1 = 'graph_1'
    graph_id2 = 'graph_2'

    x1.require_grad(graph_id1)
    x2.require_grad(graph_id2)

    xs = (x1, x2)
    expected_gxs = (xchainer.full(shape, 5, dtype), xchainer.XchainerError)

    def fprop(xs_, extra_xs_):
        x1, x2 = xs_
        y = x1 * x2
        return y,

    check_backprop(xs, expected_gxs, fprop, (), graph_id1)


def test_backward_multiple_graphs_non_existing():
    shape = (1,)
    dtype = xchainer.float32

    x1 = xchainer.full(shape, 2, dtype)
    x2 = xchainer.full(shape, 5, dtype)

    graph_id1 = 'graph_1'
    graph_id2 = 'graph_2'

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

    graph_id1 = 'graph_1'
    graph_id2 = 'graph_2'

    x1.require_grad(graph_id1)
    x2.require_grad(graph_id2)

    xs = (x1, x2)

    def fprop(xs_, extra_xs_):
        x1, x2 = xs_
        y = x1 * x2
        return y,

    expected_gxs = (xchainer.full(shape, 5, dtype), xchainer.XchainerError)
    check_backprop(xs, expected_gxs, fprop, (), graph_id1)

    x1.set_grad(None, graph_id1)
    x2.set_grad(None, graph_id2)

    expected_gxs = (xchainer.XchainerError, xchainer.full(shape, 2, dtype))
    check_backprop(xs, expected_gxs, fprop, (), graph_id2)

    x1.set_grad(None, graph_id1)
    x2.set_grad(None, graph_id2)

    x1.require_grad(graph_id2)
    x2.require_grad(graph_id1)

    expected_gxs = (xchainer.full(shape, 5, dtype), xchainer.full(shape, 2, dtype))
    check_backprop(xs, expected_gxs, fprop, (), graph_id2)

    assert x1.get_grad(graph_id1) is None
    assert x2.get_grad(graph_id1) is None
