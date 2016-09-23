from __future__ import division
import sys

import numpy
import six

try:
    from cupy import core
except ImportError:
    # core is a c-extension module.
    # When a user cannot import core, it represents that CuPy is not correctly
    # built.
    msg = ('CuPy is not correctly installed. Please check your environment, '
           'uninstall Chainer and reinstall it with `pip install chainer '
           '--no-cache-dir -vvvv`.')
    raise six.reraise(RuntimeError, RuntimeError(msg), sys.exc_info()[2])


from cupy.core import fusion
from cupy import creation
from cupy import indexing
from cupy import io
from cupy import linalg
from cupy import manipulation
import cupy.random
from cupy import sorting
from cupy import statistics
from cupy import testing  # NOQA
from cupy import util

random = cupy.random

ndarray = core.ndarray

# dtype short cut
number = numpy.number
integer = numpy.integer
signedinteger = numpy.signedinteger
unsignedinteger = numpy.unsignedinteger
inexact = numpy.inexact
floating = numpy.floating

bool_ = numpy.bool_
byte = numpy.byte
short = numpy.short
intc = numpy.intc
int_ = numpy.int_
longlong = numpy.longlong
ubyte = numpy.ubyte
ushort = numpy.ushort
uintc = numpy.uintc
uint = numpy.uint
ulonglong = numpy.ulonglong

half = numpy.half
single = numpy.single
float_ = numpy.float_
longfloat = numpy.longfloat

int8 = numpy.int8
int16 = numpy.int16
int32 = numpy.int32
int64 = numpy.int64
uint8 = numpy.uint8
uint16 = numpy.uint16
uint32 = numpy.uint32
uint64 = numpy.uint64

float16 = numpy.float16
float32 = numpy.float32
float64 = numpy.float64

ufunc = core.ufunc

newaxis = numpy.newaxis  # == None

# =============================================================================
# Routines
#
# The order of these declarations are borrowed from the NumPy document:
# http://docs.scipy.org/doc/numpy/reference/routines.html
# =============================================================================

# -----------------------------------------------------------------------------
# Array creation routines
# -----------------------------------------------------------------------------
empty = creation.basic.empty
empty_like = creation.basic.empty_like
eye = creation.basic.eye
identity = creation.basic.identity
ones = creation.basic.ones
ones_like = creation.basic.ones_like
zeros = creation.basic.zeros
zeros_like = creation.basic.zeros_like
full = creation.basic.full
full_like = creation.basic.full_like

array = creation.from_data.array
asarray = creation.from_data.asarray
asanyarray = creation.from_data.asanyarray
ascontiguousarray = creation.from_data.ascontiguousarray
copy = fusion.copy

arange = creation.ranges.arange
linspace = creation.ranges.linspace

diag = creation.matrix.diag
diagflat = creation.matrix.diagflat

# -----------------------------------------------------------------------------
# Array manipulation routines
# -----------------------------------------------------------------------------
copyto = manipulation.basic.copyto

reshape = manipulation.shape.reshape
ravel = manipulation.shape.ravel

rollaxis = manipulation.transpose.rollaxis
swapaxes = manipulation.transpose.swapaxes
transpose = manipulation.transpose.transpose

atleast_1d = manipulation.dims.atleast_1d
atleast_2d = manipulation.dims.atleast_2d
atleast_3d = manipulation.dims.atleast_3d
broadcast = manipulation.dims.broadcast
broadcast_arrays = manipulation.dims.broadcast_arrays
broadcast_to = manipulation.dims.broadcast_to
expand_dims = manipulation.dims.expand_dims
squeeze = manipulation.dims.squeeze

column_stack = manipulation.join.column_stack
concatenate = manipulation.join.concatenate
dstack = manipulation.join.dstack
hstack = manipulation.join.hstack
vstack = manipulation.join.vstack
stack = manipulation.join.stack

asfortranarray = manipulation.kind.asfortranarray

array_split = manipulation.split.array_split
dsplit = manipulation.split.dsplit
hsplit = manipulation.split.hsplit
split = manipulation.split.split
vsplit = manipulation.split.vsplit

tile = manipulation.tiling.tile
repeat = manipulation.tiling.repeat

roll = manipulation.rearrange.roll

# -----------------------------------------------------------------------------
# Binary operations
# -----------------------------------------------------------------------------
bitwise_and = fusion.bitwise_and
bitwise_or = fusion.bitwise_or
bitwise_xor = fusion.bitwise_xor
invert = fusion.invert
left_shift = fusion.left_shift
right_shift = fusion.right_shift

binary_repr = numpy.binary_repr

# -----------------------------------------------------------------------------
# Data type routines (borrowed from NumPy)
# -----------------------------------------------------------------------------
can_cast = numpy.can_cast
promote_types = numpy.promote_types
min_scalar_type = numpy.min_scalar_type
result_type = numpy.result_type
common_type = numpy.common_type
obj2sctype = numpy.obj2sctype

dtype = numpy.dtype
format_parser = numpy.format_parser

finfo = numpy.finfo
iinfo = numpy.iinfo
MachAr = numpy.MachAr

issctype = numpy.issctype
issubdtype = numpy.issubdtype
issubsctype = numpy.issubsctype
issubclass_ = numpy.issubclass_
find_common_type = numpy.find_common_type

typename = numpy.typename
sctype2char = numpy.sctype2char
mintypecode = numpy.mintypecode

# -----------------------------------------------------------------------------
# Optionally Scipy-accelerated routines
# -----------------------------------------------------------------------------
# TODO(beam2d): Implement it

# -----------------------------------------------------------------------------
# Discrete Fourier Transform
# -----------------------------------------------------------------------------
# TODO(beam2d): Implement it

# -----------------------------------------------------------------------------
# Indexing routines
# -----------------------------------------------------------------------------
take = indexing.indexing.take
diagonal = indexing.indexing.diagonal
ix_ = indexing.generate.ix_

fill_diagonal = indexing.insert.fill_diagonal
# -----------------------------------------------------------------------------
# Input and output
# -----------------------------------------------------------------------------
load = io.npz.load
save = io.npz.save
savez = io.npz.savez
savez_compressed = io.npz.savez_compressed

array_repr = io.formatting.array_repr
array_str = io.formatting.array_str

base_repr = numpy.base_repr

# -----------------------------------------------------------------------------
# Linear algebra
# -----------------------------------------------------------------------------
dot = linalg.product.dot
vdot = linalg.product.vdot
inner = linalg.product.inner
outer = linalg.product.outer
tensordot = linalg.product.tensordot

trace = linalg.norm.trace

# -----------------------------------------------------------------------------
# Logic functions
# -----------------------------------------------------------------------------
isfinite = fusion.isfinite
isinf = fusion.isinf
isnan = fusion.isnan

isscalar = numpy.isscalar

logical_and = fusion.logical_and
logical_or = fusion.logical_or
logical_not = fusion.logical_not
logical_xor = fusion.logical_xor

greater = fusion.greater
greater_equal = fusion.greater_equal
less = fusion.less
less_equal = fusion.less_equal
equal = fusion.equal
not_equal = fusion.not_equal

all = fusion.all
any = fusion.any

# -----------------------------------------------------------------------------
# Mathematical functions
# -----------------------------------------------------------------------------
sin = fusion.sin
cos = fusion.cos
tan = fusion.tan
arcsin = fusion.arcsin
arccos = fusion.arccos
arctan = fusion.arctan
hypot = fusion.hypot
arctan2 = fusion.arctan2
deg2rad = fusion.deg2rad
rad2deg = fusion.rad2deg
degrees = fusion.degrees
radians = fusion.radians

sinh = fusion.sinh
cosh = fusion.cosh
tanh = fusion.tanh
arcsinh = fusion.arcsinh
arccosh = fusion.arccosh
arctanh = fusion.arctanh

rint = fusion.rint
floor = fusion.floor
ceil = fusion.ceil
trunc = fusion.trunc

sum = fusion.sum
prod = fusion.prod

exp = fusion.exp
expm1 = fusion.expm1
exp2 = fusion.exp2
log = fusion.log
log10 = fusion.log10
log2 = fusion.log2
log1p = fusion.log1p
logaddexp = fusion.logaddexp
logaddexp2 = fusion.logaddexp2

signbit = fusion.signbit
copysign = fusion.copysign
ldexp = fusion.ldexp
frexp = fusion.frexp
nextafter = fusion.nextafter

add = fusion.add
reciprocal = fusion.reciprocal
negative = fusion.negative
multiply = fusion.multiply
divide = fusion.divide
power = fusion.power
subtract = fusion.subtract
true_divide = fusion.true_divide
floor_divide = fusion.floor_divide
fmod = fusion.fmod
mod = fusion.mod
modf = fusion.modf
remainder = fusion.remainder

clip = fusion.clip
sqrt = fusion.sqrt
square = fusion.square
absolute = fusion.absolute
abs = fusion.abs
sign = fusion.sign
maximum = fusion.maximum
minimum = fusion.minimum
fmax = fusion.fmax
fmin = fusion.fmin

# -----------------------------------------------------------------------------
# Sorting, searching, and counting
# -----------------------------------------------------------------------------
count_nonzero = sorting.count.count_nonzero
nonzero = sorting.search.nonzero
flatnonzero = sorting.search.flatnonzero

argmax = sorting.search.argmax
argmin = sorting.search.argmin
where = fusion.where

# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------
amin = fusion.amin
min = fusion.amin
amax = fusion.amax
max = fusion.amax

mean = statistics.meanvar.mean
var = statistics.meanvar.var
std = statistics.meanvar.std

bincount = statistics.histogram.bincount


# CuPy specific functions
def asnumpy(a, stream=None):
    """Returns an array on the host memory from an arbitrary source array.

    Args:
        a: Arbitrary object that can be converted to :class:`numpy.ndarray`.
        stream (cupy.cuda.Stream): CUDA stream object. If it is specified, then
            the device-to-host copy runs asynchronously. Otherwise, the copy is
            synchronous. Note that if ``a`` is not a :class:`cupy.ndarray`
            object, then this argument has no effect.

    Returns:
        numpy.ndarray: Converted array on the host memory.

    """
    if isinstance(a, ndarray):
        return a.get(stream=stream)
    else:
        return numpy.asarray(a)


_cupy = sys.modules[__name__]


def get_array_module(*args):
    """Returns the array module for arguments.

    This function is used to implement CPU/GPU generic code. If at least one of
    the arguments is a :class:`cupy.ndarray` object, the :mod:`cupy` module is
    returned.

    Args:
        args: Values to determine whether NumPy or CuPy should be used.

    Returns:
        module: :mod:`cupy` or :mod:`numpy` is returned based on the types of
        the arguments.

    .. admonition:: Example

       A NumPy/CuPy generic function can be written as follows

       >>> def softplus(x):
       ...     xp = cupy.get_array_module(x)
       ...     return xp.maximum(0, x) + xp.log1p(xp.exp(-abs(x)))

    """
    if six.moves.builtins.any(isinstance(arg, ndarray) for arg in args):
        return _cupy
    else:
        return numpy


clear_memo = util.clear_memo
memoize = util.memoize

ElementwiseKernel = core.ElementwiseKernel
ReductionKernel = core.ReductionKernel

fuse = fusion.fuse
