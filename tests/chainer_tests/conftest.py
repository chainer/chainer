import pytest

import chainerx


if not chainerx.is_available():
    # Skip all ChainerX tests if it is unavailable.
    # TODO(kmaehashi) add `not chainerx` condition to chainer-test.
    pytest.mark.chainerx = pytest.mark.skip


# testing.run_module(__name__, __file__)
