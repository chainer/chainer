import pytest

import chainerx


def _check_backward_unary(fprop):
    x = chainerx.ndarray((3,), chainerx.float32, [1., 2., 1.])
    x.require_grad()

    chainerx.check_backward(
        fprop,
        (x,),
        (chainerx.ndarray((3,), chainerx.float32, [0., -2., 1.]),),
        (chainerx.full((3,), 1e-3, chainerx.float32),),
    )


def test_correct_backward_unary():
    _check_backward_unary(lambda xs: (xs[0] * xs[0],))


def test_incorrect_backward_unary():
    # as_grad_stopped() makes backward not corresponding to the mathematical differentiation of the forward computation,
    # which should be detected by check_backward.
    def fprop(xs):
        x, = xs
        return (x * x).as_grad_stopped() + x,
    with pytest.raises(chainerx.GradientCheckError):
        _check_backward_unary(fprop)


def _check_backward_binary(fprop):
    chainerx.check_backward(
        fprop,
        (chainerx.ndarray((3,), chainerx.float32, [1., -2., 1.]).require_grad(),
         chainerx.ndarray((3,), chainerx.float32, [0., 1., 2.]).require_grad()),
        (chainerx.ndarray((3,), chainerx.float32, [1., -2., 3.]),),
        (chainerx.full((3,), 1e-3, chainerx.float32), chainerx.full((3,), 1e-3, chainerx.float32)),
    )


def test_correct_backward_binary():
    _check_backward_binary(lambda xs: (xs[0] * xs[1],))


def test_incorrect_backward_binary():
    # See the comment of test_incorrect_backward_unary().
    def fprop(xs):
        x, y = xs
        return (x * y).as_grad_stopped() + x + y,
    with pytest.raises(chainerx.GradientCheckError):
        _check_backward_binary(fprop)


def test_correct_double_backward_unary():
    chainerx.check_double_backward(
        lambda xs: (xs[0] * xs[0],),
        (chainerx.ndarray((3,), chainerx.float32, [1., 2., 3.]).require_grad(),),
        (chainerx.ones((3,), chainerx.float32).require_grad(),),
        (chainerx.ones((3,), chainerx.float32),),
        (chainerx.full((3,), 1e-3, chainerx.float32), chainerx.full((3,), 1e-3, chainerx.float32)),
        1e-4,
        1e-3,
    )


def test_correct_double_backward_binary():
    chainerx.check_double_backward(
        lambda xs: (xs[0] * xs[1],),
        (chainerx.ndarray((3,), chainerx.float32, [1., 2., 3.]).require_grad(),
         chainerx.ones((3,), chainerx.float32).require_grad()),
        (chainerx.ones((3,), chainerx.float32).require_grad(),),
        (chainerx.ones((3,), chainerx.float32), chainerx.ones((3,), chainerx.float32)),
        (chainerx.full((3,), 1e-3, chainerx.float32),
         chainerx.full((3,), 1e-3, chainerx.float32),
         chainerx.full((3,), 1e-3, chainerx.float32)),
        1e-4,
        1e-3,
    )
