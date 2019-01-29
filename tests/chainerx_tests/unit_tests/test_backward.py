import chainerx

import pytest


def _assert_arrays_equal(array1, array2):
    if array1 is None:
        assert array1 == array2
    else:
        assert array1.dtype == array2.dtype
        assert array1.shape == array2.shape
        assert array1._debug_flat_data == array2._debug_flat_data


def _check_backward(fprop, xs, expected_gxs, gys=None, backprop_id=None):
    # Checks for test validity.
    assert callable(fprop)
    assert isinstance(xs, tuple)
    assert isinstance(expected_gxs, tuple)
    assert len(xs) == len(expected_gxs)
    assert all([isinstance(a, chainerx.ndarray) for a in xs])
    assert all([(isinstance(a, chainerx.ndarray) or a ==
                 chainerx.ChainerxError) for a in expected_gxs])

    # Forward.
    ys = fprop(*xs)

    # Set output gradients.
    if gys is not None:
        assert len(gys) == len(ys)
        for y, gy in zip(ys, gys):
            assert not y.is_grad_required()
            y.set_grad(gy, backprop_id)

    # Backward.
    chainerx.backward(ys, backprop_id)

    # Check gradients of input arrays.
    for x, expected_gx in zip(xs, expected_gxs):
        if expected_gx is chainerx.ChainerxError:
            with pytest.raises(chainerx.ChainerxError):
                x.get_grad(backprop_id)
        else:
            gx = x.get_grad(backprop_id)
            _assert_arrays_equal(gx, expected_gx)

    # Check gradients of output arrays.
    if gys is None:
        gys = (None,) * len(xs)
    for y, gy in zip(ys, gys):
        if gy is None:
            assert not y.is_grad_required(backprop_id)
            with pytest.raises(chainerx.ChainerxError):
                y.get_grad(backprop_id)
        else:
            assert y.is_grad_required(backprop_id)
            _assert_arrays_equal(gy, y.get_grad(backprop_id))


def _check_grad(
        fprop, xs, expected_gxs, gys=None, backprop_id=None, xs_indices=None,
        ys_indices=None):
    # Checks for test validity.
    assert callable(fprop)
    assert isinstance(xs, tuple)
    assert isinstance(expected_gxs, tuple)
    assert len(xs) == len(expected_gxs)
    assert all([isinstance(a, chainerx.ndarray) for a in xs])
    assert all([(isinstance(a, chainerx.ndarray) or a ==
                 chainerx.ChainerxError) for a in expected_gxs])

    # Forward.
    ys = fprop(*xs)

    # Set output gradients.
    if gys is not None:
        assert len(gys) == len(ys)
        for y, gy in zip(ys, gys):
            assert not y.is_grad_required()
            y.set_grad(gy, backprop_id)

    # Backward using grad.
    initial_gxs = []
    for x in xs:
        if x.is_grad_required():
            initial_gxs.append(x.get_grad(backprop_id))
        else:
            initial_gxs.append(chainerx.ChainerxError)

    if xs_indices is not None:
        actual_xs = tuple([xs[i] for i in xs_indices])
    else:
        actual_xs = xs
    if ys_indices is not None:
        actual_ys = tuple([ys[i] for i in ys_indices])
    else:
        actual_ys = ys
    gxs = chainerx.grad(actual_ys, actual_xs, backprop_id)

    # Check gradients.
    for gx, expected_gx in zip(gxs, expected_gxs):
        _assert_arrays_equal(gx, expected_gx)

    # Check gradients of output arrays.
    if gys is None:
        gys = (None,) * len(xs)
    for y, gy in zip(ys, gys):
        if gy is None:
            assert not y.is_grad_required(backprop_id)
            with pytest.raises(chainerx.ChainerxError):
                y.get_grad(backprop_id)
        else:
            assert y.is_grad_required(backprop_id)
            _assert_arrays_equal(gy, y.get_grad(backprop_id))

    # Check initial gradients of inputs and that they are not modified.
    for x, initial_gx in zip(xs, initial_gxs):
        if initial_gx is chainerx.ChainerxError:
            assert not x.is_grad_required(backprop_id)
            with pytest.raises(chainerx.ChainerxError):
                x.get_grad(backprop_id)
        else:
            assert x.is_grad_required(backprop_id)
            _assert_arrays_equal(initial_gx, x.get_grad(backprop_id))


def _identity(x):
    return x.copy(),


def _add(x0, x1):
    return x0 + x1,


def _add_identical_input(x0):
    return x0 + x0,


def _mul(x0, x1):
    return x0 * x1,


def _add_mul(x0, x1, x2):
    return x0 * (x1 + x2),


def _add_mul_identical_input_to_multiple_ops(x0, x1):
    return x0 * (x0 + x1),


def _add_identical_intermediate_input(x0):
    h = x0 + x0
    return h + h,


def _binary_math_multiple_outputs(x1, x2):
    return (x1 + x2, x1 * x2)


def _backward_with_double_backprop(x0, x1):
    assert x0.is_grad_required()

    h = x0 * (x0 + x1)
    chainerx.backward(h, enable_double_backprop=True)
    gx0 = x0.get_grad()
    x0.cleargrad()
    return gx0,


def test_backward_identity():
    shape = (1,)
    dtype = chainerx.float32

    xs = (chainerx.full(shape, 5, dtype).require_grad(),)
    expected_gxs = (chainerx.full(shape, 1, dtype),)

    _check_backward(_identity, xs, expected_gxs)


def test_grad_identity():
    shape = (1,)
    dtype = chainerx.float32

    xs = (chainerx.full(shape, 5, dtype).require_grad(),)
    expected_gxs = (chainerx.full(shape, 1, dtype),)

    _check_grad(_identity, xs, expected_gxs)


def test_backward_add():
    shape = (1,)
    dtype = chainerx.float32

    xs = (
        chainerx.full(shape, 3, dtype).require_grad(),
        chainerx.full(shape, 5, dtype).require_grad(),)
    expected_gxs = (
        chainerx.full(shape, 1, dtype),
        chainerx.full(shape, 1, dtype),)

    _check_backward(_add, xs, expected_gxs)


def test_backward_mul():
    shape = (1,)
    dtype = chainerx.float32

    xs = (
        chainerx.full(shape, 3, dtype).require_grad(),
        chainerx.full(shape, 5, dtype).require_grad(),)
    expected_gxs = (
        chainerx.full(shape, 5, dtype),
        chainerx.full(shape, 3, dtype),)

    _check_backward(_mul, xs, expected_gxs)


def test_backward_add_mull():
    shape = (1,)
    dtype = chainerx.float32

    xs = (
        chainerx.full(shape, 2, dtype).require_grad(),
        chainerx.full(shape, 9, dtype).require_grad(),
        chainerx.full(shape, 5, dtype).require_grad(),)
    expected_gxs = (
        chainerx.full(shape, 14, dtype),
        chainerx.full(shape, 2, dtype),
        chainerx.full(shape, 2, dtype))

    _check_backward(_add_mul, xs, expected_gxs)


def test_backward_add_mul_extra_inputs():
    shape = (1,)
    dtype = chainerx.float32

    xs = (
        chainerx.full(shape, 3, dtype).require_grad(),
        chainerx.full(shape, 2, dtype).require_grad(),
        chainerx.full(shape, 4, dtype))
    expected_gxs = (
        chainerx.full(shape, 6, dtype),
        chainerx.full(shape, 3, dtype),
        chainerx.ChainerxError)

    _check_backward(_add_mul, xs, expected_gxs)


def test_double_backprop():
    shape = (1,)
    dtype = chainerx.float32

    xs = (
        chainerx.full(shape, 2, dtype).require_grad(),
        chainerx.full(shape, 3, dtype),)
    expected_gxs = (
        chainerx.full(shape, 2, dtype),
        chainerx.ChainerxError,)

    _check_backward(_backward_with_double_backprop, xs, expected_gxs)


def test_multiple_graphs_double_backprop():
    shape = (1,)
    dtype = chainerx.float32

    with chainerx.backprop_scope('bp_x1') as bp_x1, \
            chainerx.backprop_scope('bp_x0') as bp_x0:
        xs = (
            chainerx.full(shape, 2, dtype).require_grad(bp_x0),
            chainerx.full(shape, 3, dtype).require_grad(bp_x1),)
        expected_gxs = (
            chainerx.ChainerxError,
            chainerx.full(shape, 2, dtype),)

        def fprop(x0, x1):
            assert x0.is_grad_required(bp_x0)

            h = x0 * (x0 + x1)
            chainerx.backward(h, backprop_id=bp_x0)
            gx0 = x0.get_grad(bp_x0)  # 2x + h

            assert not gx0.is_backprop_required(bp_x0)
            assert gx0.is_backprop_required(bp_x1)

            return x0 * gx0,

        _check_backward(fprop, xs, expected_gxs, backprop_id=bp_x1)


def test_backward_identical_input_to_multiple_ops():
    shape = (1,)
    dtype = chainerx.float32

    xs = (
        chainerx.full(shape, 2, dtype).require_grad(),
        chainerx.full(shape, 3, dtype),)
    expected_gxs = (
        chainerx.full(shape, 7, dtype),
        chainerx.ChainerxError,)

    _check_backward(_add_mul_identical_input_to_multiple_ops, xs, expected_gxs)


def test_backward_identical_inputs():
    shape = (1,)
    dtype = chainerx.float32

    xs = (chainerx.full(shape, 2, dtype).require_grad(),)
    expected_gxs = (chainerx.full(shape, 2, dtype),)

    _check_backward(_add_identical_input, xs, expected_gxs)


def test_backward_identical_intermediate_nodes():
    shape = (1,)
    dtype = chainerx.float32

    xs = (chainerx.full(shape, 2, dtype).require_grad(),)
    expected_gxs = (chainerx.full(shape, 4, dtype),)

    _check_backward(_add_identical_intermediate_input, xs, expected_gxs)


def test_backward_given_input_grad():
    shape = (1,)
    dtype = chainerx.float32

    xs = (chainerx.full(shape, 1, dtype).require_grad(),)
    expected_gxs = (chainerx.full(shape, 2, dtype),)

    def fprop(x):
        x.set_grad(chainerx.full(shape, 1, dtype))
        return x.copy(),

    _check_backward(fprop, xs, expected_gxs)


def test_backward_given_output_grad():
    shape = (1,)
    dtype = chainerx.float32

    xs = (
        chainerx.full(shape, 2, dtype).require_grad(),
        chainerx.full(shape, 3, dtype),)
    expected_gxs = (
        chainerx.full(shape, 6, dtype),
        chainerx.ChainerxError,)
    gys = (
        chainerx.full(shape, 2, dtype),)

    _check_backward(_mul, xs, expected_gxs, gys=gys)


def test_backward_multiple_outputs():
    shape = (1,)
    dtype = chainerx.float32

    xs = (
        chainerx.full(shape, 3, dtype).require_grad(),
        chainerx.full(shape, 5, dtype).require_grad(),)
    expected_gxs = (
        chainerx.full(shape, 6, dtype),
        chainerx.full(shape, 4, dtype),)

    _check_backward(_binary_math_multiple_outputs, xs, expected_gxs)


def test_backward_multiple_graphs_basic():
    shape = (1,)
    dtype = chainerx.float32

    with chainerx.backprop_scope('bp1') as backprop_id1, \
            chainerx.backprop_scope('bp2') as backprop_id2:
        xs = (
            chainerx.full(shape, 2, dtype).require_grad(backprop_id1),
            chainerx.full(shape, 5, dtype).require_grad(backprop_id2),)
        expected_gxs = (
            chainerx.full(shape, 5, dtype),
            chainerx.ChainerxError,)

        _check_backward(_mul, xs, expected_gxs, backprop_id=backprop_id1)


def test_backward_multiple_graphs_reuse():
    shape = (1,)
    dtype = chainerx.float32

    with chainerx.backprop_scope('bp2') as backprop_id2, \
            chainerx.backprop_scope('bp1') as backprop_id1:
        xs = (
            chainerx.full(shape, 2, dtype).require_grad(backprop_id1),
            chainerx.full(shape, 5, dtype).require_grad(backprop_id2),)
        expected_gxs = (
            chainerx.full(shape, 5, dtype),
            chainerx.ChainerxError,)

        _check_backward(_mul, xs, expected_gxs, backprop_id=backprop_id1)

        x1, x2 = xs
        x1.cleargrad(backprop_id1)
        x2.cleargrad(backprop_id2)

        assert x1.get_grad(backprop_id1) is None
        assert x2.get_grad(backprop_id2) is None

        expected_gxs = (
            chainerx.ChainerxError,
            chainerx.full(shape, 2, dtype),)

        _check_backward(_mul, xs, expected_gxs, backprop_id=backprop_id2)

        x1.cleargrad(backprop_id1)
        x2.cleargrad(backprop_id2)

        x1.require_grad(backprop_id2)
        x2.require_grad(backprop_id1)

        expected_gxs = (
            chainerx.full(shape, 5, dtype),
            chainerx.full(shape, 2, dtype),)

        _check_backward(_mul, xs, expected_gxs, backprop_id=backprop_id2)

        assert x1.get_grad(backprop_id1) is None
        assert x2.get_grad(backprop_id1) is None


def test_backward_sole_array_node():
    shape = (1,)
    dtype = chainerx.float32

    x = chainerx.full(shape, 2, dtype).require_grad()
    expected_gx = chainerx.full(shape, 1, dtype)

    chainerx.backward(x)

    _assert_arrays_equal(x.get_grad(), expected_gx)


def test_backward_keyword_arguments():
    shape = (1,)
    dtype = chainerx.float32

    with chainerx.backprop_scope('bp1') as backprop_id1:
        x = chainerx.full(shape, 2, dtype).require_grad(backprop_id1)

        chainerx.backward(x, backprop_id=backprop_id1)

        with pytest.raises(
                TypeError, match=r'.*incompatible function arguments.*'):
            chainerx.backward(body=x, backprop_id=backprop_id1)


def test_create_and_release_backprop_id():
    context = chainerx.Context()
    backprop_id = context.make_backprop_id("bp1")

    assert "bp1" == backprop_id.name
    assert context == backprop_id.context

    context._check_valid_backprop_id(backprop_id)

    context.release_backprop_id(backprop_id)

    with pytest.raises(chainerx.ChainerxError):
        context._check_valid_backprop_id(backprop_id)


def test_backward_multiple_graphs_non_existing():
    shape = (1,)
    dtype = chainerx.float32

    with chainerx.backprop_scope('bp1') as backprop_id1, \
            chainerx.backprop_scope('bp2') as backprop_id2:
        xs = (
            chainerx.full(shape, 2, dtype).require_grad(backprop_id1),
            chainerx.full(shape, 5, dtype).require_grad(backprop_id1),)

        y = xs[0] * xs[1]

        with pytest.raises(chainerx.ChainerxError):
            chainerx.backward(y, backprop_id2)
