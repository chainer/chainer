from chainer.testing.array import assert_allclose  # NOQA
from chainer.testing.backend import BackendConfig  # NOQA
from chainer.testing.backend import inject_backend_tests  # NOQA
from chainer.testing.distribution_test import distribution_unittest  # NOQA
from chainer.testing.function_link import FunctionTestCase  # NOQA
from chainer.testing.function_link import FunctionTestError  # NOQA
from chainer.testing.function_link import InitializerArgument  # NOQA
from chainer.testing.function_link import LinkInitializersTestCase  # NOQA
from chainer.testing.function_link import LinkTestCase  # NOQA
from chainer.testing.function_link import LinkTestError  # NOQA
from chainer.testing.helper import assert_warns  # NOQA
from chainer.testing.helper import patch  # NOQA
from chainer.testing.helper import with_requires  # NOQA
from chainer.testing.helper import without_requires  # NOQA
from chainer.testing.parameterized import parameterize  # NOQA
from chainer.testing.parameterized import parameterize_pytest  # NOQA
from chainer.testing.parameterized import product  # NOQA
from chainer.testing.parameterized import product_dict  # NOQA
from chainer.testing.random import fix_random  # NOQA
from chainer.testing.random import generate_seed  # NOQA
from chainer.testing.serializer import save_and_load  # NOQA
from chainer.testing.serializer import save_and_load_hdf5  # NOQA
from chainer.testing.serializer import save_and_load_npz  # NOQA
from chainer.testing.training import get_trainer_with_mock_updater  # NOQA
from chainer.testing.unary_math_function_test import unary_math_function_unittest  # NOQA


def run_module(name, file):
    """Run current test cases of the file.

    Args:
        name: __name__ attribute of the file.
        file: __file__ attribute of the file.
    """

    if name == '__main__':
        import pytest
        pytest.main([file, '-vvs', '-x', '--pdb'])
