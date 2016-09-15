import nose

from chainer.testing import array
from chainer.testing import helper
from chainer.testing import parameterized


assert_allclose = array.assert_allclose

parameterize = parameterized.parameterize
product = parameterized.product
product_dict = parameterized.product_dict

with_requires = helper.with_requires


def run_module(name, file):
    """Run current test cases of the file.

    Args:
        name: __name__ attribute of the file.
        file: __file__ attribute of the file.
    """

    if name == '__main__':

        nose.runmodule(argv=[file, '-vvs', '-x', '--pdb', '--pdb-failure'],
                       exit=False)
