import pytest

pytest.register_assert_rewrite('xchainer.testing.array')
pytest.register_assert_rewrite('xchainer.testing.helper')

from xchainer.testing import array  # NOQA
from xchainer.testing import helper  # NOQA

from xchainer.testing.array import assert_array_equal  # NOQA
from xchainer.testing.helper import numpy_xchainer_array_equal  # NOQA
