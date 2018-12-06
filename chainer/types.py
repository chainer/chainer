from typing import Any  # NOQA
from typing import Optional  # NOQA
from typing import Sequence  # NOQA
from typing import Tuple  # NOQA
from typing import Union  # NOQA
from typing_extensions import Protocol  # NOQA

import numbers

try:
    from typing import TYPE_CHECKING  # NOQA
except ImportError:
    # typing.TYPE_CHECKING doesn't exist before Python 3.5.2
    TYPE_CHECKING = False

# import chainer modules only for type checkers to avoid circular import
if TYPE_CHECKING:
    from types import ModuleType  # NOQA

    import numpy  # NOQA

    from chainer.backend import Device  # NOQA
    from chainer.backends import cuda, intel64  # NOQA
    from chainer import initializer  # NOQA

    import chainerx  # NOQA


Shape = Tuple[int, ...]


ShapeSpec = Union[int, Sequence[int]]  # Sequence includes Tuple[int, ...]


DTypeSpec = Union[Any]  # TODO(okapies): encode numpy.dtype


NdArray = Union[
    'numpy.ndarray',
    'cuda.ndarray',
    # 'intel64.mdarray',
    # TODO(okapies): mdarray is partially imcompatible with other ndarrays
    'chainerx.ndarray',
]
"""The ndarray types which are compatible with :func:`chainer.get_array_types`
"""


Xp = Union[Any]  # TODO(okapies): encode numpy/cupy/chainerx


class AbstractInitializer(Protocol):
    """Protocol class for Initializer.

    It can be either an :class:`chainer.Initializer` or a callable object
    that takes an ndarray.

    This is only for PEP 544 compatible static type checkers.
    """
    dtype = None  # type: Optional[DTypeSpec]

    def __call__(self, array):
        # type: (NdArray) -> None
        pass


ScalarValue = Union[
    'numpy.generic',
    bytes,
    str,
    memoryview,
    numbers.Number,
]
"""The scalar types which are compatible with :func:`numpy.isscalar`.
"""


InitializerSpec = Union[AbstractInitializer, ScalarValue, 'numpy.ndarray']


DeviceSpec = Union[
    'Device',
    'chainerx.Device',  # See https://github.com/python/mypy/issues/5908 # NOQA
    'cuda.Device',
    str,
    Tuple[str, int],
    'ModuleType',
    Tuple['ModuleType', int],
]


CudaDeviceSpec = Union['cuda.Device', int, 'numpy.integer']  # NOQA
"""
This type only for the deprecated :func:`chainer.cuda.get_device` API.
Use :class:`~chainer.types.DeviceSpec` instead.
"""
