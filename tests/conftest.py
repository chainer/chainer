import os

from chainer import testing
from chainer.testing import parameterized


_pairwise_parameterize = (
    os.environ.get('CHAINER_TEST_PAIRWISE_PARAMETERIZATION', 'never'))
assert _pairwise_parameterize in ('never', 'always')


def pytest_collection(session):
    # Perform pairwise testing.
    # TODO(kataoka): This is a tentative fix. Discuss its public interface.
    if _pairwise_parameterize == 'always':
        pairwise_product_dict = parameterized._pairwise_product_dict
        testing.product_dict = pairwise_product_dict
        parameterized.product_dict = pairwise_product_dict


def pytest_collection_finish(session):
    if _pairwise_parameterize == 'always':
        product_dict = parameterized._product_dict_orig
        testing.product_dict = product_dict
        parameterized.product_dict = product_dict
