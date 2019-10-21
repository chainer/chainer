import pytest

from chainer.backends import cuda
import chainerx


if not chainerx.is_available():
    # Skip all ChainerX tests if ChainerX is unavailable.
    # TODO(kmaehashi) This is an tentative fix. This file should be removed
    # once chainer-test supports ChainerX.
    pytest.mark.chainerx = pytest.mark.skip


def pytest_runtest_teardown(item, nextitem):
    if cuda.available:
        assert cuda.cupy.cuda.runtime.getDevice() == 0


# testing.run_module(__name__, __file__)
