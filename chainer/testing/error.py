import contextlib

from chainer import utils


class TestError(AssertionError):

    """Parent class to Chainer test errors.

    .. seealso::
        :class:`chainer.testing.FunctionTestError`
        :class:`chainer.testing.LinkTestError`

    """

    @classmethod
    def check(cls, expr, message):
        if not expr:
            raise cls(message)

    @classmethod
    def fail(cls, message, exc=None):
        if exc is not None:
            utils._raise_from(cls, message, exc)
        raise cls(message)

    @classmethod
    @contextlib.contextmanager
    def raise_if_fail(cls, message, error_types=AssertionError):
        try:
            yield
        except error_types as e:
            cls.fail(message, e)
