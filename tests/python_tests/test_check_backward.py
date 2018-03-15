import pytest

import xchainer


def _check_backward_unary(fprop):
    xchainer.check_backward(
        fprop,
        (xchainer.Array((3,), xchainer.float32, [1., 2., 1.]).require_grad(),),
        (xchainer.Array((3,), xchainer.float32, [0., -2., 1.]),),
        (xchainer.full((3,), 1e-3, xchainer.float32),),
    )


def test_correct_backward_unary():
    _check_backward_unary(lambda xs: (xs[0] * xs[0],))


@pytest.mark.xfail
def test_incorrect_backward_unary():
    def fprop(xs):
        x, = xs
        return (x * x).as_constant() + x,
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


@pytest.mark.xfail
def test_incorrect_backward_binary():
    def fprop(xs):
        x, y = xs
        return (x * y).as_constant() + x + y,
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
