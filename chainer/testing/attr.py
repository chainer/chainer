from functools import wraps
import os
import unittest

from chainer import numexpr_config
from chainer import testing


try:
    import pytest
    _error = None
except ImportError as e:
    _error = e


def is_available():
    return _error is None


def check_available():
    if _error is not None:
        raise RuntimeError('''\
{} is not available.

Reason: {}: {}'''.format(__name__, type(_error).__name__, _error))


def get_error():
    return _error


if _error is None:
    _gpu_limit = int(os.getenv('CHAINER_TEST_GPU_LIMIT', '-1'))

    cudnn = pytest.mark.cudnn
    slow = pytest.mark.slow

else:
    def _dummy_callable(*args, **kwargs):
        check_available()
        assert False  # Not reachable

    cudnn = _dummy_callable
    slow = _dummy_callable


def multi_gpu(gpu_num):
    """Decorator to indicate number of GPUs required to run the test.

    Tests can be annotated with this decorator (e.g., ``@multi_gpu(2)``) to
    declare number of GPUs required to run. When running tests, if
    ``CHAINER_TEST_GPU_LIMIT`` environment variable is set to value greater
    than or equals to 0, test cases that require GPUs more than the limit will
    be skipped.
    """

    check_available()
    return unittest.skipIf(
        0 <= _gpu_limit and _gpu_limit < gpu_num,
        reason='{} GPUs required'.format(gpu_num))


def gpu(f):
    """Decorator to indicate that GPU is required to run the test.

    Tests can be annotated with this decorator (e.g., ``@gpu``) to
    declare that one GPU is required to run.
    """

    check_available()
    return multi_gpu(1)(pytest.mark.gpu(f))


def no_numexpr(f):
    """Decorator to indicate that the test is for non-numxepr code

    Tests can be annotated with this decorator to declare that numexpr is not
    used to run
    """
    @wraps(f)
    def wrapper(arg):
        ne_enabled = numexpr_config.numexpr_enabled
        numexpr_config.numexpr_enabled = False
        f(arg)
        numexpr_config.numexpr_enabled = ne_enabled
    return wrapper


def with_numexpr(f):
    """Decorator to indicate that the test is for numexpr code

    Tests should be annotated with this decorator to declare that numexpr
    is required to run them
    """
    @wraps(f)
    @testing.with_requires('numexpr')
    def wrapper(arg):
        ne_enabled = True  # NOQA
        f(arg)
    return wrapper
