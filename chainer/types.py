import numbers
import typing as tp  # NOQA
import typing_extensions as tpe  # NOQA

try:
    from typing import TYPE_CHECKING  # NOQA
except ImportError:
    # typing.TYPE_CHECKING doesn't exist before Python 3.5.2
    TYPE_CHECKING = False

# import chainer modules only for type checkers to avoid circular import
if TYPE_CHECKING:
    from types import ModuleType  # NOQA

    import numpy  # NOQA

    from chainer import backend  # NOQA
    from chainer.backends import cuda, intel64  # NOQA
    from chainer import initializer  # NOQA

    import chainerx  # NOQA


Shape = tp.Tuple[int, ...]


ShapeSpec = tp.Union[int, tp.Sequence[int]]  # Sequence includes Tuple[int, ...] # NOQA


DTypeSpec = tp.Union[tp.Any]  # TODO(okapies): encode numpy.dtype


NdArray = tp.Union[
    'numpy.ndarray',
    'cuda.ndarray',
    # 'intel64.mdarray',
    # TODO(okapies): mdarray is partially incompatible with other ndarrays
    'chainerx.ndarray',
]
"""The ndarray types supported in :func:`chainer.get_array_types`
"""


class Dataset(tpe.Protocol):
    """Protocol class for dataset.

    This is only for PEP 544 compliant static type checkers.
    """

    def __getitem__(self, index):
        pass

    def __len__(self):
        # type: () -> int
        pass


Datasets = tp.Union[
    tp.Sequence[Dataset],  # Sequence includes both List and Tuple
    NdArray,
    tp.Dict[tp.Any, Dataset],
]


Xp = tp.Union[tp.Any]  # TODO(okapies): encode numpy/cupy/ideep/chainerx


class AbstractInitializer(tpe.Protocol):
    """Protocol class for Initializer.

    It can be either an :class:`chainer.Initializer` or a callable object
    that takes an ndarray.

    This is only for PEP 544 compliant static type checkers.
    """
    dtype = None  # type: tp.Optional[DTypeSpec]

    def __call__(self, array):
        # type: (NdArray) -> None
        pass


ScalarValue = tp.Union[
    'numpy.generic',
    bytes,
    str,
    memoryview,
    numbers.Number,
]
"""The scalar types supported in :func:`numpy.isscalar`.
"""


PaddingSpec = tp.Union[
    ScalarValue,
    tp.Tuple[ScalarValue, ...],
    tp.Dict[tp.Any, ScalarValue]
]


InitializerSpec = tp.Union[AbstractInitializer, ScalarValue, 'numpy.ndarray']


DeviceSpec = tp.Union[
    'backend.Device',
    'chainerx.Device',
    'cuda.Device',
    str,
    tp.Tuple[str, int],
    'ModuleType',  # numpy and intel64 module
    tp.Tuple['ModuleType', int],  # cupy module and device ID
]
"""The device specifier types supported in :func:`chainer.get_device`
"""
# TODO(okapies): Use Xp instead of ModuleType


CudaDeviceSpec = tp.Union['cuda.Device', int, 'numpy.integer']  # NOQA
"""
This type only for the deprecated :func:`chainer.cuda.get_device` API.
Use :class:`~chainer.types.DeviceSpec` instead.
"""
