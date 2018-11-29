import chainerx

import pytest


def _assert_arrays_equal(array1, array2):
    if array1 is None:
        assert array1 == array2
    else:
        assert array1.dtype == array2.dtype
        assert array1.shape == array2.shape
        assert array1._debug_flat_data == array2._debug_flat_data


def _check_backprop(
        xs, expected_gxs, fprop, extra_xs, gys=None, backprop_id=None):
    # Checks for test validity
    assert isinstance(xs, tuple)
    assert isinstance(expected_gxs, tuple)
    assert callable(fprop)
    assert isinstance(extra_xs, tuple)
    assert len(xs) == len(expected_gxs)
    assert all([isinstance(a, chainerx.ndarray) for a in xs])
    assert all([(isinstance(a, chainerx.ndarray) or a ==
                 chainerx.ChainerxError) for a in expected_gxs])
    assert all([isinstance(a, chainerx.ndarray) for a in extra_xs])

    # Forward
    outputs = fprop(xs, extra_xs)

    # Set output gradients
    if gys is None:
        gys = (None,) * len(outputs)
    assert len(gys) == len(outputs)
    for output, gy in zip(outputs, gys):
        assert not output.is_grad_required()
        output.set_grad(gy, backprop_id)

    # Backward
    chainerx.backward(outputs, backprop_id)

    # Check gradients of input arrays
    for i, expected_gx in enumerate(expected_gxs):
        x = xs[i]
        if expected_gx is chainerx.ChainerxError:
            with pytest.raises(chainerx.ChainerxError):
                x.get_grad(backprop_id)
        else:
            gx = x.get_grad(backprop_id)
            _assert_arrays_equal(gx, expected_gx)

    # Check gradients of output arrays
    for output, gy in zip(outputs, gys):
        if gy is None:
            assert not output.is_grad_required(backprop_id)
            with pytest.raises(chainerx.ChainerxError):
                output.get_grad(backprop_id)
        else:
            assert output.is_grad_required(backprop_id)
            _assert_arrays_equal(gy, output.get_grad(backprop_id))


def test_backward_identity():
    shape = (1,)
    dtype = chainerx.float32

    xs = (chainerx.full(shape, 5, dtype),)
    expected_gxs = (chainerx.full(shape, 1, dtype),)

    for x in xs:
        x.require_grad()

    def fprop(xs_, extra_xs_):
        x, = xs_
        y = x.copy()
        return y,

    _check_backprop(xs, expected_gxs, fprop, ())


def test_backward_add():
    shape = (1,)
    dtype = chainerx.float32

    xs = (
        chainerx.full(shape, 3, dtype),
        chainerx.full(shape, 5, dtype),)
    expected_gxs = (
        chainerx.full(shape, 1, dtype),
        chainerx.full(shape, 1, dtype),)

    for x in xs:
        x.require_grad()

    def fprop(xs_, extra_xs_):
        x0, x1 = xs_
        y = x0 + x1
        return y,

    _check_backprop(xs, expected_gxs, fprop, ())


def test_backward_mul():
    shape = (1,)
    dtype = chainerx.float32

    xs = (
        chainerx.full(shape, 3, dtype),
        chainerx.full(shape, 5, dtype),)
    expected_gxs = (
        chainerx.full(shape, 5, dtype),
        chainerx.full(shape, 3, dtype),)

    for x in xs:
        x.require_grad()

    def fprop(xs_, extra_xs_):
        x0, x1 = xs_
        y = x0 * x1
        return y,

    _check_backprop(xs, expected_gxs, fprop, ())


def test_backward_add_mull():
    shape = (1,)
    dtype = chainerx.float32

    xs = (
        chainerx.full(shape, 2, dtype),
        chainerx.full(shape, 9, dtype),
        chainerx.full(shape, 5, dtype),)
    expected_gxs = (
        chainerx.full(shape, 14, dtype),
        chainerx.full(shape, 2, dtype),
        chainerx.full(shape, 2, dtype))

    for x in xs:
        x.require_grad()

    def fprop(xs_, extra_xs_):
        x0, x1, x2 = xs_
        y = x0 * (x1 + x2)
        return y,

    _check_backprop(xs, expected_gxs, fprop, ())


def test_backward_add_mul_extra_inputs():
    shape = (1,)
    dtype = chainerx.float32

    xs = (
        chainerx.full(shape, 2, dtype),
        chainerx.full(shape, 3, dtype))
    extra_xs = (chainerx.full(shape, 4, dtype),)
    expected_gxs = (
        chainerx.full(shape, 3, dtype),
        chainerx.full(shape, 6, dtype))

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
    dtype = chainerx.float32

    x = chainerx.full(shape, 2, dtype)
    expected_gx = chainerx.full(shape, 1, dtype)

    x.require_grad()

    chainerx.backward(x)

    _assert_arrays_equal(x.get_grad(), expected_gx)


def test_double_backprop():
    shape = (1,)
    dtype = chainerx.float32

    xs = (chainerx.full(shape, 2, dtype),)
    extra_xs = (chainerx.full(shape, 3, dtype),)
    expected_gxs = (chainerx.full(shape, 2, dtype),)

    for x in xs:
        x.require_grad()

    def fprop(xs_, extra_xs_):
        x, = xs_
        t, = extra_xs_
        y = x * (x + t)
        chainerx.backward(y, enable_double_backprop=True)
        gx = x.get_grad()  # 2x + y
        x.cleargrad()
        return gx,

    _check_backprop(xs, expected_gxs, fprop, extra_xs)


def test_multiple_graphs_double_backprop():
    with chainerx.backprop_scope('bp_y') as bp_y, \
            chainerx.backprop_scope('bp_x') as bp_x:

        x = chainerx.full((1,), 2, chainerx.float32)
        x.require_grad(backprop_id=bp_x)

        y = chainerx.full((1,), 3, chainerx.float32)
        y.require_grad(backprop_id=bp_y)

        z = x * (x + y)
        chainerx.backward(z, backprop_id=bp_x)

        gx = x.get_grad(bp_x)  # 2x + y
        assert not gx.is_backprop_required(backprop_id=bp_x)
        assert gx.is_backprop_required(backprop_id=bp_y)

        w = x * gx
        chainerx.backward(w, backprop_id=bp_y)

        e = chainerx.full((1,), 2, chainerx.float32)

        _assert_arrays_equal(y.get_grad(bp_y), e)  # x


def test_backward_input_to_multiple_ops():
    shape = (1,)
    dtype = chainerx.float32

    xs = (chainerx.full(shape, 2, dtype),)
    extra_xs = (chainerx.full(shape, 3, dtype),)
    expected_gxs = (chainerx.full(shape, 7, dtype),)

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
    dtype = chainerx.float32

    xs = (chainerx.full(shape, 2, dtype),)
    expected_gxs = (chainerx.full(shape, 2, dtype),)

    for x in xs:
        x.require_grad()

    def fprop(xs_, extra_xs_):
        x, = xs_
        y = x + x
        return y,

    _check_backprop(xs, expected_gxs, fprop, ())


def test_backward_identical_intermediate_nodes():
    shape = (1,)
    dtype = chainerx.float32

    xs = (chainerx.full(shape, 2, dtype),)
    expected_gxs = (chainerx.full(shape, 4, dtype),)

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
    dtype = chainerx.float32

    xs = (chainerx.full(shape, 1, dtype),)
    expected_gxs = (chainerx.full(shape, 2, dtype),)

    for x in xs:
        x.require_grad()

    def fprop(xs_, extra_xs_):
        x, = xs_
        x.set_grad(chainerx.full(shape, 1, dtype))
        y = x.copy()
        return y,

    _check_backprop(xs, expected_gxs, fprop, ())


def test_backward_given_output_grad():
    shape = (1,)
    dtype = chainerx.float32

    xs = (chainerx.full(shape, 2, dtype),)
    extra_xs = (chainerx.full(shape, 3, dtype),)
    expected_gxs = (chainerx.full(shape, 6, dtype),)
    gys = (chainerx.full(shape, 2, dtype),)

    for x in xs:
        x.require_grad()

    def fprop(xs_, extra_xs_):
        x, = xs_
        t, = extra_xs_
        y = x * t
        return y,

    _check_backprop(xs, expected_gxs, fprop, extra_xs, gys)


def test_backward_keyword_arguments():
    x = chainerx.full((1,), 2, chainerx.float32)
    with chainerx.backprop_scope('bp1') as backprop_id1:
        x.require_grad(backprop_id=backprop_id1)
        chainerx.backward(x, backprop_id=backprop_id1)
        with pytest.raises(
                TypeError, match=r'.*incompatible function arguments.*'):
            chainerx.backward(body=x, backprop_id=backprop_id1)


def test_backward_multiple_graphs_basic():
    shape = (1,)
    dtype = chainerx.float32

    x1 = chainerx.full(shape, 2, dtype)
    x2 = chainerx.full(shape, 5, dtype)

    with chainerx.backprop_scope('bp1') as backprop_id1, \
            chainerx.backprop_scope('bp2') as backprop_id2:

        x1.require_grad(backprop_id1)
        x2.require_grad(backprop_id2)

        xs = (x1, x2)
        expected_gxs = (chainerx.full(shape, 5, dtype), chainerx.ChainerxError)

        def fprop(xs_, extra_xs_):
            x1, x2 = xs_
            y = x1 * x2
            return y,

        _check_backprop(xs, expected_gxs, fprop, (), backprop_id=backprop_id1)


def test_backward_multiple_graphs_non_existing():
    shape = (1,)
    dtype = chainerx.float32

    x1 = chainerx.full(shape, 2, dtype)
    x2 = chainerx.full(shape, 5, dtype)

    with chainerx.backprop_scope('bp1') as backprop_id1, \
            chainerx.backprop_scope('bp2') as backprop_id2:

        x1.require_grad(backprop_id1)
        x2.require_grad(backprop_id1)

        y = x1 * x2
        with pytest.raises(chainerx.ChainerxError):
            chainerx.backward(y, backprop_id2)


def test_backward_multiple_graphs_reuse():
    shape = (1,)
    dtype = chainerx.float32

    x1 = chainerx.full(shape, 2, dtype)
    x2 = chainerx.full(shape, 5, dtype)

    with chainerx.backprop_scope('bp2') as backprop_id2, \
            chainerx.backprop_scope('bp1') as backprop_id1:

        x1.require_grad(backprop_id1)
        x2.require_grad(backprop_id2)

        xs = (x1, x2)

        def fprop(xs_, extra_xs_):
            x1, x2 = xs_
            y = x1 * x2
            return y,

        expected_gxs = (chainerx.full(shape, 5, dtype), chainerx.ChainerxError)
        _check_backprop(xs, expected_gxs, fprop, (), backprop_id=backprop_id1)

        x1.cleargrad(backprop_id1)
        x2.cleargrad(backprop_id2)

        assert x1.get_grad(backprop_id1) is None
        assert x2.get_grad(backprop_id2) is None

        expected_gxs = (chainerx.ChainerxError, chainerx.full(shape, 2, dtype))
        _check_backprop(xs, expected_gxs, fprop, (), backprop_id=backprop_id2)

        x1.cleargrad(backprop_id1)
        x2.cleargrad(backprop_id2)

        x1.require_grad(backprop_id2)
        x2.require_grad(backprop_id1)

        expected_gxs = (chainerx.full(shape, 5, dtype),
                        chainerx.full(shape, 2, dtype))
        _check_backprop(xs, expected_gxs, fprop, (), backprop_id=backprop_id2)

        assert x1.get_grad(backprop_id1) is None
        assert x2.get_grad(backprop_id1) is None


def test_backward_multiple_outputs():
    shape = (1,)
    dtype = chainerx.float32

    xs = (
        chainerx.full(shape, 3, dtype),
        chainerx.full(shape, 5, dtype),)
    expected_gxs = (
        chainerx.full(shape, 6, dtype),
        chainerx.full(shape, 4, dtype),)

    for x in xs:
        x.require_grad()

    def fprop(xs_, extra_xs_):
        x0, x1 = xs_
        return (x0 + x1, x0 * x1)

    _check_backprop(xs, expected_gxs, fprop, ())


def test_create_and_release_backprop_id():
    context = chainerx.Context()
    backprop_id = context.make_backprop_id("bp1")
    assert "bp1" == backprop_id.name
    assert context == backprop_id.context
    context._check_valid_backprop_id(backprop_id)

    context.release_backprop_id(backprop_id)
    with pytest.raises(chainerx.ChainerxError):
        context._check_valid_backprop_id(backprop_id)
