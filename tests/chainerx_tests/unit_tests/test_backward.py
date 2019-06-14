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
    assert all([isinstance(a, chainerx.ndarray) or a is None
                for a in expected_gxs])

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
        if expected_gx is None:
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
    assert all([isinstance(a, chainerx.ndarray) for a in xs])
    assert all([isinstance(a, chainerx.ndarray) or a is None
                for a in expected_gxs])

    # Forward.
    ys = fprop(*xs)

    # Set output gradients.
    if gys is not None:
        assert len(gys) == len(ys)
        for y, gy in zip(ys, gys):
            assert not y.is_grad_required()
            y.set_grad(gy, backprop_id)

    # Backward using grad.
    initial_gxs = [
        x.get_grad(backprop_id) if x.is_grad_required(backprop_id)
        else chainerx.ChainerxError for x in xs]

    if xs_indices is not None:
        actual_xs = tuple([xs[i] for i in xs_indices])
        assert len(actual_xs) == len(expected_gxs)
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


# Delegates work to either _check_backward or _check_grad.
def _check_backprop(
        method, fprop, xs, expected_gxs, gys=None, backprop_id=None):
    if method == 'backward':
        check_func = _check_backward
    elif method == 'grad':
        check_func = _check_grad
    else:
        assert False

    check_func(fprop, xs, expected_gxs, gys=gys, backprop_id=backprop_id)


def parametrize_backprop(argname='method'):
    return pytest.mark.parametrize(argname, ['backward', 'grad'])


@parametrize_backprop()
def test_backprop_identity(method):
    shape = (1,)
    dtype = chainerx.float32

    xs = (chainerx.full(shape, 5, dtype).require_grad(),)
    expected_gxs = (chainerx.full(shape, 1, dtype),)

    def fprop(x):
        return x.copy(),

    _check_backprop(method, fprop, xs, expected_gxs)


@parametrize_backprop()
def test_backprop_add(method):
    shape = (1,)
    dtype = chainerx.float32

    xs = (
        chainerx.full(shape, 3, dtype).require_grad(),
        chainerx.full(shape, 5, dtype).require_grad(),)
    expected_gxs = (
        chainerx.full(shape, 1, dtype),
        chainerx.full(shape, 1, dtype),)

    def fprop(x0, x1):
        return x0 + x1,

    _check_backprop(method, fprop, xs, expected_gxs)


@parametrize_backprop()
def test_backprop_mul(method):
    shape = (1,)
    dtype = chainerx.float32

    xs = (
        chainerx.full(shape, 3, dtype).require_grad(),
        chainerx.full(shape, 5, dtype).require_grad(),)
    expected_gxs = (
        chainerx.full(shape, 5, dtype),
        chainerx.full(shape, 3, dtype),)

    def fprop(x0, x1):
        return x0 * x1,

    _check_backprop(method, fprop, xs, expected_gxs)


@parametrize_backprop()
def test_backprop_add_mul(method):
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

    def fprop(x0, x1, x2):
        return x0 * (x1 + x2),

    _check_backprop(method, fprop, xs, expected_gxs)


@parametrize_backprop()
def test_backprop_add_mul_extra_inputs(method):
    shape = (1,)
    dtype = chainerx.float32

    xs = (
        chainerx.full(shape, 2, dtype).require_grad(),
        chainerx.full(shape, 3, dtype).require_grad(),
        chainerx.full(shape, 4, dtype))
    expected_gxs = (
        chainerx.full(shape, 7, dtype),
        chainerx.full(shape, 2, dtype),
        None)

    def fprop(x0, x1, x2):
        return x0 * (x1 + x2),

    _check_backprop(method, fprop, xs, expected_gxs)


@parametrize_backprop()
def test_backprop_sole_array_node(method):
    shape = (1,)
    dtype = chainerx.float32

    x = chainerx.full(shape, 2, dtype).require_grad()
    expected_gx = chainerx.full(shape, 1, dtype)

    if method == 'backward':
        chainerx.backward(x)
        gx = x.get_grad()
    elif method == 'grad':
        gx, = chainerx.grad([x], [x])
    else:
        assert False

    _assert_arrays_equal(gx, expected_gx)


@parametrize_backprop()
def test_backprop_double_backprop(method):
    shape = (1,)
    dtype = chainerx.float32

    xs = (
        chainerx.full(shape, 2, dtype).require_grad(),
        chainerx.full(shape, 3, dtype),)
    expected_gxs = (
        chainerx.full(shape, 2, dtype),
        None,)

    def fprop(x0, x1):
        assert x0.is_grad_required()

        h = x0 * (x0 + x1)
        chainerx.backward(h, enable_double_backprop=True)
        gx0 = x0.get_grad()
        x0.cleargrad()
        return gx0,

    _check_backprop(method, fprop, xs, expected_gxs)


@parametrize_backprop('method0')
@parametrize_backprop('method1')
def test_backprop_multiple_graphs_double_backprop(method0, method1):
    shape = (1,)
    dtype = chainerx.float32

    with chainerx.backprop_scope('bp_x1') as bp_x1, \
            chainerx.backprop_scope('bp_x0') as bp_x0:
        xs = (
            chainerx.full(shape, 2, dtype).require_grad(bp_x0),
            chainerx.full(shape, 3, dtype).require_grad(bp_x1),)
        expected_gxs = (
            None,
            chainerx.full(shape, 2, dtype),)

        def fprop(x0, x1):
            assert x0.is_grad_required(bp_x0)

            h = x0 * (x0 + x1)
            if method0 == 'backward':
                chainerx.backward(h, backprop_id=bp_x0)
                gx0 = x0.get_grad(bp_x0)
            elif method0 == 'grad':
                gx0, = chainerx.grad([h], [x0], backprop_id=bp_x0)
            else:
                assert False

            assert not gx0.is_backprop_required(bp_x0)
            assert gx0.is_backprop_required(bp_x1)

            return x0 * gx0,

        _check_backprop(method1, fprop, xs, expected_gxs, backprop_id=bp_x1)


@parametrize_backprop()
def test_backprop_identical_input_to_multiple_ops(method):
    shape = (1,)
    dtype = chainerx.float32

    xs = (
        chainerx.full(shape, 2, dtype).require_grad(),
        chainerx.full(shape, 3, dtype),)
    expected_gxs = (
        chainerx.full(shape, 7, dtype),
        None,)

    def fprop(x0, x1):
        return x0 * (x0 + x1),

    _check_backprop(method, fprop, xs, expected_gxs)


@parametrize_backprop()
def test_backprop_identical_inputs(method):
    shape = (1,)
    dtype = chainerx.float32

    xs = (chainerx.full(shape, 2, dtype).require_grad(),)
    expected_gxs = (chainerx.full(shape, 2, dtype),)

    def fprop(x):
        return x + x,

    _check_backprop(method, fprop, xs, expected_gxs)


@parametrize_backprop()
def test_backprop_identical_intermediate_nodes(method):
    shape = (1,)
    dtype = chainerx.float32

    xs = (chainerx.full(shape, 2, dtype).require_grad(),)
    expected_gxs = (chainerx.full(shape, 4, dtype),)

    def fprop(x):
        h = x + x
        return h + h,

    _check_backprop(method, fprop, xs, expected_gxs)


@parametrize_backprop()
def test_backprop_given_input_grad(method):
    shape = (1,)
    dtype = chainerx.float32

    xs = (chainerx.full(shape, 1, dtype).require_grad(),)
    expected_gx_value = 2 if method == 'backward' else 1
    expected_gxs = (chainerx.full(shape, expected_gx_value, dtype),)

    def fprop(x):
        x.set_grad(chainerx.full(shape, 1, dtype))
        return x.copy(),

    _check_backprop(method, fprop, xs, expected_gxs)


@parametrize_backprop()
def test_backprop_given_output_grad(method):
    shape = (1,)
    dtype = chainerx.float32

    xs = (
        chainerx.full(shape, 2, dtype).require_grad(),
        chainerx.full(shape, 3, dtype),)
    expected_gxs = (
        chainerx.full(shape, 6, dtype),
        None,)
    gys = (
        chainerx.full(shape, 2, dtype),)

    def fprop(x0, x1):
        return x0 * x1,

    _check_backprop(method, fprop, xs, expected_gxs, gys=gys)


@parametrize_backprop()
def test_backprop_multiple_graphs_basic(method):
    shape = (1,)
    dtype = chainerx.float32

    with chainerx.backprop_scope('bp1') as backprop_id1, \
            chainerx.backprop_scope('bp2') as backprop_id2:
        xs = (
            chainerx.full(shape, 2, dtype).require_grad(backprop_id1),
            chainerx.full(shape, 5, dtype).require_grad(backprop_id2),)
        expected_gxs = (
            chainerx.full(shape, 5, dtype),
            None,)

        def fprop(x0, x1):
            return x0 * x1,

        _check_backprop(
            method, fprop, xs, expected_gxs, backprop_id=backprop_id1)


@parametrize_backprop()
def test_backprop_multiple_graphs_non_existing(method):
    shape = (1,)
    dtype = chainerx.float32

    with chainerx.backprop_scope('bp1') as backprop_id1, \
            chainerx.backprop_scope('bp2') as backprop_id2:
        xs = (
            chainerx.full(shape, 2, dtype).require_grad(backprop_id1),
            chainerx.full(shape, 5, dtype).require_grad(backprop_id1),)

        y = xs[0] * xs[1]

        if method == 'backward':
            chainerx.backward(y, backprop_id2)
            assert xs[0].get_grad(backprop_id1) is None
            assert xs[1].get_grad(backprop_id1) is None
        elif method == 'grad':
            grads = chainerx.grad([y], xs, backprop_id2)
            assert len(grads) == 2
            assert grads[0] is None
            assert grads[1] is None
        else:
            assert False

        with pytest.raises(chainerx.ChainerxError):
            xs[0].get_grad(backprop_id2)
        with pytest.raises(chainerx.ChainerxError):
            xs[1].get_grad(backprop_id2)


@parametrize_backprop('method0')
@parametrize_backprop('method1')
@parametrize_backprop('method2')
def test_backprop_multiple_graphs_reuse(method0, method1, method2):
    shape = (1,)
    dtype = chainerx.float32

    def fprop(x0, x1):
        return x0 * x1,

    with chainerx.backprop_scope('bp2') as backprop_id2, \
            chainerx.backprop_scope('bp1') as backprop_id1:
        xs = (
            chainerx.full(shape, 2, dtype).require_grad(backprop_id1),
            chainerx.full(shape, 5, dtype).require_grad(backprop_id2),)
        expected_gxs = (
            chainerx.full(shape, 5, dtype),
            None,)

        _check_backprop(
            method0, fprop, xs, expected_gxs, backprop_id=backprop_id1)

        x1, x2 = xs
        x1.cleargrad(backprop_id1)
        x2.cleargrad(backprop_id2)

        assert x1.get_grad(backprop_id1) is None
        assert x2.get_grad(backprop_id2) is None

        expected_gxs = (
            None,
            chainerx.full(shape, 2, dtype),)

        _check_backprop(
            method1, fprop, xs, expected_gxs, backprop_id=backprop_id2)

        x1.cleargrad(backprop_id1)
        x2.cleargrad(backprop_id2)

        x1.require_grad(backprop_id2)
        x2.require_grad(backprop_id1)

        expected_gxs = (
            chainerx.full(shape, 5, dtype),
            chainerx.full(shape, 2, dtype),)

        _check_backprop(
            method2, fprop, xs, expected_gxs, backprop_id=backprop_id2)

        assert x1.get_grad(backprop_id1) is None
        assert x2.get_grad(backprop_id1) is None


@parametrize_backprop()
def test_backprop_multiple_outputs(method):
    shape = (1,)
    dtype = chainerx.float32

    xs = (
        chainerx.full(shape, 3, dtype).require_grad(),
        chainerx.full(shape, 5, dtype).require_grad(),)
    expected_gxs = (
        chainerx.full(shape, 6, dtype),
        chainerx.full(shape, 4, dtype),)

    def fprop(x0, x1):
        return x0 + x1, x0 * x1

    _check_backprop(method, fprop, xs, expected_gxs)


def test_create_and_release_backprop_id():
    context = chainerx.Context()
    backprop_id = context.make_backprop_id('bp1')

    assert 'bp1' == backprop_id.name
    assert context == backprop_id.context

    context._check_valid_backprop_id(backprop_id)

    context.release_backprop_id(backprop_id)

    with pytest.raises(chainerx.ChainerxError):
        context._check_valid_backprop_id(backprop_id)


@pytest.mark.parametrize('xs_indices', [[], [0], [1], [0, 1], [1, 0]])
@pytest.mark.parametrize('ys_indices', [[], [0], [1], [0, 1], [1, 0]])
def test_grad_not_all_inputs_outputs_in_graph(xs_indices, ys_indices):
    shape = (1,)
    dtype = chainerx.float32

    xs = (
        chainerx.full(shape, 3, dtype).require_grad(),
        chainerx.full(shape, 5, dtype).require_grad(),)
    gxs = (
        (chainerx.full(shape, 1, dtype),  # gy1gx1
         chainerx.full(shape, 1, dtype)),  # gy1gx2
        (chainerx.full(shape, 5, dtype),  # gy2gx1
         chainerx.full(shape, 3, dtype)),)  # gy2gx2
    expected_gxs = [None] * len(xs_indices)

    for ys_index in ys_indices:
        for i, xs_index in enumerate(xs_indices):
            if expected_gxs[i] is None:
                expected_gxs[i] = chainerx.full(shape, 0, dtype)
            expected_gxs[i] += gxs[ys_index][xs_index]

    def fprop(x0, x1):
        return x0 + x1, x0 * x1

    _check_grad(
        fprop, xs, tuple(expected_gxs), xs_indices=xs_indices,
        ys_indices=ys_indices)
