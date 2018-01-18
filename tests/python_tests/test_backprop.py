import xchainer


def full(shape, value, dtype):
    return xchainer.full(shape, value, dtype)


def assert_arrays_equal(array1, array2):
    assert array1.dtype == array2.dtype
    assert array1.shape == array2.shape
    assert array1._debug_flat_data == array2._debug_flat_data


def check_backprop(xs, expected_gxs, fprop, args):
    # Checks for test validity
    assert isinstance(xs, tuple)
    assert isinstance(expected_gxs, tuple)
    assert callable(fprop)
    assert isinstance(args, tuple)
    assert len(xs) == len(expected_gxs)
    assert all([isinstance(a, xchainer.Array) for a in xs])
    assert all([isinstance(a, xchainer.Array) for a in expected_gxs])
    assert all([isinstance(a, xchainer.Array) for a in args])

    outputs = fprop(xs, args)
    assert len(outputs) == 1, 'This test does not support multi-output functions yet'

    xchainer.backward(outputs[0])

    gxs = tuple([x.grad for x in xs])
    # Note: i for pytest output on failure
    for i, (gx, expected_gx) in enumerate(zip(gxs, expected_gxs)):
        assert_arrays_equal(gx, expected_gx)


def test_backward_identity():
    shape = (1,)
    dtype = xchainer.float32

    xs = (
        full(shape, 5, dtype),)
    expected_gxs = (
        full(shape, 1, dtype),)

    for x in xs:
        x.requires_grad = True

    def fprop(xs_, extra_xs_):
        x, = xs_
        y = x
        return y,

    check_backprop(xs, expected_gxs, fprop, ())


def test_backward_add():
    shape = (1,)
    dtype = xchainer.float32

    xs = (
        full(shape, 3, dtype),
        full(shape, 5, dtype),)
    expected_gxs = (
        full(shape, 1, dtype),
        full(shape, 1, dtype),)

    for x in xs:
        x.requires_grad = True

    def fprop(xs_, extra_xs_):
        x0, x1 = xs_
        y = x0 + x1
        return y,

    check_backprop(xs, expected_gxs, fprop, ())


def test_backward_mul():
    shape = (1,)
    dtype = xchainer.float32

    xs = (
        full(shape, 3, dtype),
        full(shape, 5, dtype),)
    expected_gxs = (
        full(shape, 5, dtype),
        full(shape, 3, dtype),)

    for x in xs:
        x.requires_grad = True

    def fprop(xs_, extra_xs_):
        x0, x1 = xs_
        y = x0 * x1
        return y,

    check_backprop(xs, expected_gxs, fprop, ())


def test_backward_add_mull():
    shape = (1,)
    dtype = xchainer.float32

    xs = (
        full(shape, 2, dtype),
        full(shape, 9, dtype),
        full(shape, 5, dtype),)
    expected_gxs = (
        full(shape, 14, dtype),
        full(shape, 2, dtype),
        full(shape, 2, dtype))

    for x in xs:
        x.requires_grad = True

    def fprop(xs_, extra_xs_):
        x0, x1, x2 = xs_
        y = x0 * (x1 + x2)
        return y,

    check_backprop(xs, expected_gxs, fprop, ())


def test_backward_add_mul_extra_inputs():
    shape = (1,)
    dtype = xchainer.float32

    xs = (
        full(shape, 2, dtype),
        full(shape, 3, dtype))
    extra_xs = (
        full(shape, 4, dtype),)
    expected_gxs = (
        full(shape, 3, dtype),
        full(shape, 6, dtype))

    for x in xs:
        x.requires_grad = True

    def fprop(xs_, extra_xs_):
        x0, x1 = xs_
        t0, = extra_xs_
        y = x1 * (x0 + t0)
        return y,

    check_backprop(xs, expected_gxs, fprop, extra_xs)


def test_backward_sole_array_node():
    shape = (1,)
    dtype = xchainer.float32

    x = full(shape, 2, dtype)
    expected_gx = full(shape, 1, dtype)

    x.requires_grad = True

    xchainer.backward(x)

    assert_arrays_equal(x.grad, expected_gx)


def test_double_backprop():
    shape = (1,)
    dtype = xchainer.float32

    xs = (
        full(shape, 2, dtype),)
    extra_xs = (
        full(shape, 3, dtype),)
    expected_gxs = (
        full(shape, 2, dtype),)

    for x in xs:
        x.requires_grad = True

    def fprop(xs_, extra_xs_):
        x, = xs_
        t, = extra_xs_
        y = x * (x + t)
        xchainer.backward(y)
        gx = x.grad
        x.grad = None
        return gx,

    check_backprop(xs, expected_gxs, fprop, extra_xs)


def test_backward_input_to_multiple_ops():
    shape = (1,)
    dtype = xchainer.float32

    xs = (
        full(shape, 2, dtype),)
    extra_xs = (
        full(shape, 3, dtype),)
    expected_gxs = (
        full(shape, 7, dtype),)

    for x in xs:
        x.requires_grad = True

    def fprop(xs_, extra_xs_):
        x, = xs_
        t, = extra_xs_
        y = x * (x + t)
        return y,

    check_backprop(xs, expected_gxs, fprop, extra_xs)


def test_backward_identical_inputs():
    shape = (1,)
    dtype = xchainer.float32

    xs = (
        full(shape, 2, dtype),)
    expected_gxs = (
        full(shape, 2, dtype),)

    for x in xs:
        x.requires_grad = True

    def fprop(xs_, extra_xs_):
        x, = xs_
        y = x + x
        return y,

    check_backprop(xs, expected_gxs, fprop, ())


def test_backward_given_input_grad():
    shape = (1,)
    dtype = xchainer.float32

    xs = (
        full(shape, 1, dtype),)
    expected_gxs = (
        full(shape, 2, dtype),)

    for x in xs:
        x.requires_grad = True

    def fprop(xs_, extra_xs_):
        x, = xs_
        x.grad = full(shape, 1, dtype)
        y = x.copy()
        return y,

    check_backprop(xs, expected_gxs, fprop, ())


def test_backward_given_output_grad():
    shape = (1,)
    dtype = xchainer.float32

    xs = (
        full(shape, 2, dtype),)
    extra_xs = (
        full(shape, 3, dtype),)
    expected_gxs = (
        full(shape, 6, dtype),)

    for x in xs:
        x.requires_grad = True

    def fprop(xs_, extra_xs_):
        x, = xs_
        t, = extra_xs_
        y = x * t
        y.grad = full(shape, 2, dtype)
        return y,

    check_backprop(xs, expected_gxs, fprop, extra_xs)
