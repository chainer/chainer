import pytest

pytest.register_assert_rewrite('xchainer.testing.array')
pytest.register_assert_rewrite('xchainer.testing.helper')

from xchainer.testing import array  # NOQA
from xchainer.testing import helper  # NOQA

from xchainer.testing.array import assert_allclose  # NOQA
from xchainer.testing.array import assert_allclose_ex  # NOQA
from xchainer.testing.array import assert_array_equal  # NOQA
from xchainer.testing.array import assert_array_equal_ex  # NOQA
from xchainer.testing.dtypes import all_dtypes  # NOQA
from xchainer.testing.dtypes import float_dtypes  # NOQA
from xchainer.testing.dtypes import nonfloat_dtypes  # NOQA
from xchainer.testing.dtypes import numeric_dtypes  # NOQA
from xchainer.testing.dtypes import parametrize_dtype_specifier  # NOQA
from xchainer.testing.dtypes import signed_dtypes  # NOQA
from xchainer.testing.dtypes import unsigned_dtypes  # NOQA
from xchainer.testing.helper import ignore  # NOQA
from xchainer.testing.helper import numpy_xchainer_allclose  # NOQA
from xchainer.testing.helper import numpy_xchainer_array_equal  # NOQA
