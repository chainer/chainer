import pytest

from chainer.backends import cuda
from chainer import testing
from chainer.testing import parameterized
import chainerx


if not chainerx.is_available():
    # Skip all ChainerX tests if ChainerX is unavailable.
    # TODO(kmaehashi) This is an tentative fix. This file should be removed
    # once chainer-test supports ChainerX.
    pytest.mark.chainerx = pytest.mark.skip


def pytest_collection(session):
    # Perform pairwise testing.
    # TODO(kataoka): This is a tentative fix. Discuss its public interface.
    pairwise_product_dict = parameterized._pairwise_product_dict
    testing.product_dict = pairwise_product_dict
    parameterized.product_dict = pairwise_product_dict


def pytest_collection_finish(session):
    product_dict = parameterized._product_dict_orig
    testing.product_dict = product_dict
    parameterized.product_dict = product_dict


def pytest_runtest_teardown(item, nextitem):
    if cuda.available:
        assert cuda.cupy.cuda.runtime.getDevice() == 0


# testing.run_module(__name__, __file__)
