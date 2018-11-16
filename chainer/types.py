from typing import Any
from typing import Callable
from typing import Sequence
from typing import SupportsInt
from typing import Tuple
from typing import Union

import numpy  # NOQA

try:
    from typing import TYPE_CHECKING  # NOQA
except ImportError as e:
    # typing.TYPE_CHECKING doesn't exist before Python 3.5.2
    TYPE_CHECKING = False

# import chainer modules only for type checkers to avoid circular import
if TYPE_CHECKING:
    from chainer.backends import cuda  # NOQA
    from chainer import initializer  # NOQA
    from chainer import variable  # NOQA

IntLike = Union[int, str, bytes, SupportsInt]

Shape = Tuple[int, ...]

ShapeLike = Union[Shape, int, Sequence[int]]

DTypeLike = Union[Any]
# TODO(okapies): DTypeLike should be concrete type(s) which expresses numpy.dtype # NOQA

NdArray = Union[numpy.ndarray, 'cuda.ndarray']

InitializerLike = Union[
    'initializer.Initializer',
    NdArray,
    Callable[[NdArray], None]
]

VariableLike = Union['variable.Variable', NdArray]

DeviceLike = Union['cuda.Device', int, 'cuda.ndarray']
