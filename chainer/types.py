from typing import Any
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from typing_extensions import Protocol

import numbers
from types import ModuleType  # NOQA

try:
    from typing import TYPE_CHECKING  # NOQA
except ImportError:
    # typing.TYPE_CHECKING doesn't exist before Python 3.5.2
    TYPE_CHECKING = False

# import chainer modules only for type checkers to avoid circular import
if TYPE_CHECKING:
    import numpy  # NOQA

    from chainer.backend import Device  # NOQA
    from chainer.backends import cuda  # NOQA
    from chainer import initializer  # NOQA

    import chainerx  # NOQA


Shape = Tuple[int, ...]


ShapeSpec = Union[int, Sequence[int]]  # Sequence includes Tuple[int, ...]


DTypeSpec = Union[Any]  # TODO(okapies): encode numpy.dtype


NdArray = Union[
    'numpy.ndarray',
    'cuda.ndarray',
    'chainerx.ndarray',
]


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


# see numpy.isscalar()
InitializerSpec = Union[
    AbstractInitializer,
    numbers.Number,
    bytes,
    str,
    memoryview,
    'numpy.generic',
    'numpy.ndarray',
]


DeviceSpec = Union[
    'Device',
    'chainerx.Device',  # See https://github.com/python/mypy/issues/5908 # NOQA
    'cuda.Device',
    str,
    Tuple[str, int],
    ModuleType,
    Tuple[ModuleType, int],
]


CudaDeviceSpec = Union['cuda.Device', int, 'numpy.integer']  # NOQA
"""
This type only for the deprecated :func:`chainer.cuda.get_device` API.
Use :class:`chainer.types.DeviceSpec` instead.
"""
