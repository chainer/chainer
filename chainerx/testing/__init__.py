import pytest

pytest.register_assert_rewrite('chainerx.testing.array')
pytest.register_assert_rewrite('chainerx.testing.helper')

from chainerx._testing import _DeviceBuffer  # NOQA
from chainerx._testing import _fromnumpy  # NOQA

from chainerx.testing import array  # NOQA
from chainerx.testing import helper  # NOQA

from chainerx.testing.array import assert_allclose  # NOQA
from chainerx.testing.array import assert_allclose_ex  # NOQA
from chainerx.testing.array import assert_array_equal  # NOQA
from chainerx.testing.array import assert_array_equal_ex  # NOQA
from chainerx.testing.dtypes import all_dtypes  # NOQA
from chainerx.testing.dtypes import float_dtypes  # NOQA
from chainerx.testing.dtypes import integral_dtypes  # NOQA
from chainerx.testing.dtypes import signed_integral_dtypes  # NOQA
from chainerx.testing.dtypes import nonfloat_dtypes  # NOQA
from chainerx.testing.dtypes import numeric_dtypes  # NOQA
from chainerx.testing.dtypes import parametrize_dtype_specifier  # NOQA
from chainerx.testing.dtypes import signed_dtypes  # NOQA
from chainerx.testing.dtypes import unsigned_dtypes  # NOQA
from chainerx.testing.helper import ignore  # NOQA
from chainerx.testing.helper import numpy_chainerx_allclose  # NOQA
from chainerx.testing.helper import numpy_chainerx_array_equal  # NOQA
