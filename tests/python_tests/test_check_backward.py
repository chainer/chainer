import pytest

import xchainer


def _check_backward_unary(fprop, requires_grad=True):
    x = xchainer.Array((3,), xchainer.float32, [1., 2., 1.])
    if requires_grad:
        x.require_grad()

    xchainer.check_backward(
        fprop,
        (x,),
        (xchainer.Array((3,), xchainer.float32, [0., -2., 1.]),),
        (xchainer.full((3,), 1e-3, xchainer.float32),),
    )


def test_correct_backward_unary():
    _check_backward_unary(lambda xs: (xs[0] * xs[0],))


def test_no_grad_required():
    _check_backward_unary(lambda xs: (xs[0] * xs[0],), False)


def test_incorrect_backward_unary():
    # as_constant() makes backward not corresponding to the mathematical differentiation of the forward computation,
    # which should be detected by check_backward.
    def fprop(xs):
        x, = xs
        return (x * x).as_constant() + x,
    with pytest.raises(xchainer.GradientCheckError):
        _check_backward_unary(fprop)


def _check_backward_binary(fprop):
    xchainer.check_backward(
        fprop,
        (xchainer.Array((3,), xchainer.float32, [1., -2., 1.]).require_grad(),
         xchainer.Array((3,), xchainer.float32, [0., 1., 2.]).require_grad()),
        (xchainer.Array((3,), xchainer.float32, [1., -2., 3.]),),
        (xchainer.full((3,), 1e-3, xchainer.float32), xchainer.full((3,), 1e-3, xchainer.float32)),
    )


def test_correct_backward_binary():
    _check_backward_binary(lambda xs: (xs[0] * xs[1],))


def test_incorrect_backward_binary():
    # See the comment of test_incorrect_backward_unary().
    def fprop(xs):
        x, y = xs
        return (x * y).as_constant() + x + y,
    with pytest.raises(xchainer.GradientCheckError):
        _check_backward_binary(fprop)


def test_correct_double_backward_unary():
    xchainer.check_double_backward(
        lambda xs: (xs[0] * xs[0],),
        (xchainer.Array((3,), xchainer.float32, [1., 2., 3.]).require_grad(),),
        (xchainer.ones((3,), xchainer.float32).require_grad(),),
        (xchainer.ones((3,), xchainer.float32),),
        (xchainer.full((3,), 1e-3, xchainer.float32), xchainer.full((3,), 1e-3, xchainer.float32)),
        1e-4,
        1e-3,
    )


def test_correct_double_backward_binary():
    xchainer.check_double_backward(
        lambda xs: (xs[0] * xs[1],),
        (xchainer.Array((3,), xchainer.float32, [1., 2., 3.]).require_grad(),
         xchainer.ones((3,), xchainer.float32).require_grad()),
        (xchainer.ones((3,), xchainer.float32).require_grad(),),
        (xchainer.ones((3,), xchainer.float32), xchainer.ones((3,), xchainer.float32)),
        (xchainer.full((3,), 1e-3, xchainer.float32),
         xchainer.full((3,), 1e-3, xchainer.float32),
         xchainer.full((3,), 1e-3, xchainer.float32)),
        1e-4,
        1e-3,
    )
