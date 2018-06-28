# This is a NumPy/CuPy-compatible namespace of Chainer. All the functions act
# on chainer.Variable instead of ndarrays. Function names are coming from
# NumPy, so sometimes the terminology is different from Chainer's one. For
# example, "array" in this namespace means "variable" in Chainer.

# =============================================================================
# NumPy/CuPy-compatible APIs
# =============================================================================

# ======== Dtypes (borrowed from NumPy) ========

# NOTE: Dtypes that are commented out are currently not supported by Chainer.

# Generic types
# from numpy import complexfloating  # NOQA
from numpy import floating  # NOQA
from numpy import generic  # NOQA
from numpy import inexact  # NOQA
from numpy import integer  # NOQA
from numpy import number  # NOQA
from numpy import signedinteger  # NOQA
# from numpy import unsignedinteger

# Bool
from numpy import bool8  # NOQA
from numpy import bool_  # NOQA

# Integers
# from numpy import byte
# from numpy import int8
# from numpy import int16
from numpy import int32  # NOQA
from numpy import int_  # NOQA
from numpy import intc  # NOQA
from numpy import intp  # NOQA
from numpy import longlong  # NOQA
# from numpy import short

# Unsigned integers
# from numpy import ubyte
# from numpy import ushort
# from numpy import uintc
# from numpy import uint
# from numpy import ulonglong
# from numpy import uintp
# from numpy import uint8
# from numpy import uint16
# from numpy import uint32
# from numpy import uint64

# Floating-point numbers
from numpy import double  # NOQA
from numpy import float16  # NOQA
from numpy import float32  # NOQA
from numpy import float64  # NOQA
from numpy import float_  # NOQA
# from numpy import float96
# from numpy import float128
from numpy import half  # NOQA
# from numpy import longfloat
from numpy import single  # NOQA

# Complex floating-point numbers
# from numpy import csingle
# from numpy import complex64
# from numpy import complex128
# from numpy import complex192
# from numpy import complex256
# from numpy import complex_
# from numpy import clongfloat

# Built-in Python types
from numpy import bool  # NOQA
from numpy import float  # NOQA
from numpy import int  # NOQA

# ======== Routines ========

# Variable creation routines
from chainer.np.creation import empty  # NOQA
from chainer.np.creation import empty_like  # NOQA
from chainer.np.creation import eye  # NOQA
from chainer.np.creation import full  # NOQA
from chainer.np.creation import full_like  # NOQA
from chainer.np.creation import identity  # NOQA
from chainer.np.creation import ones  # NOQA
from chainer.np.creation import ones_like  # NOQA
from chainer.np.creation import zeros  # NOQA
from chainer.np.creation import zeros_like  # NOQA

# Manipulation routines
from chainer.functions import reshape  # NOQA

from chainer.functions import moveaxis  # NOQA
from chainer.functions import rollaxis  # NOQA
from chainer.functions import swapaxes  # NOQA
from chainer.functions import transpose  # NOQA

from chainer.functions import broadcast_to  # NOQA
from chainer.functions import expand_dims  # NOQA
from chainer.functions import squeeze  # NOQA

from chainer.functions import dstack  # NOQA
from chainer.functions import hstack  # NOQA
from chainer.functions import stack  # NOQA
from chainer.functions import vstack  # NOQA

from chainer.functions import repeat  # NOQA
from chainer.functions import tile  # NOQA

from chainer.functions import flip  # NOQA
from chainer.functions import fliplr  # NOQA
from chainer.functions import flipud  # NOQA

# Data type routines (borrowed from NumPy)
from numpy import can_cast  # NOQA
from numpy import common_type  # NOQA
from numpy import min_scalar_type  # NOQA
from numpy import obj2sctype  # NOQA
from numpy import promote_types  # NOQA
from numpy import result_type  # NOQA

from numpy import dtype  # NOQA
from numpy import format_parser  # NOQA

from numpy import finfo  # NOQA
from numpy import iinfo  # NOQA
from numpy import MachAr  # NOQA

from numpy import find_common_type  # NOQA
from numpy import issctype  # NOQA
from numpy import issubclass_  # NOQA
from numpy import issubdtype  # NOQA
from numpy import issubsctype  # NOQA

from numpy import mintypecode  # NOQA
from numpy import sctype2char  # NOQA
from numpy import typename  # NOQA

# Linear algebra
from chainer.functions import matmul  # NOQA
from chainer.functions import tensordot  # NOQA

# Mathematical functions
from chainer.functions import arccos  # NOQA
from chainer.functions import arcsin  # NOQA
from chainer.functions import arctan  # NOQA
from chainer.functions import arctan2  # NOQA
from chainer.functions import cos  # NOQA
from chainer.functions import sin  # NOQA
from chainer.functions import tan  # NOQA

from chainer.functions import cosh  # NOQA
from chainer.functions import sinh  # NOQA
from chainer.functions import tanh  # NOQA

from chainer.functions import ceil  # NOQA
from chainer.functions import fix  # NOQA
from chainer.functions import floor  # NOQA

from chainer.functions import cumsum  # NOQA
from chainer.functions import prod  # NOQA
from chainer.functions import sum  # NOQA

from chainer.functions import exp  # NOQA
from chainer.functions import expm1  # NOQA
from chainer.functions import log  # NOQA
from chainer.functions import log10  # NOQA
from chainer.functions import log1p  # NOQA
from chainer.functions import log2  # NOQA

from chainer.functions import add  # NOQA
from chainer.functions import fmod  # NOQA

from chainer.functions import absolute as abs  # NOQA
from chainer.functions import absolute  # NOQA
from chainer.functions import clip  # NOQA
from chainer.functions import maximum  # NOQA
from chainer.functions import minimum  # NOQA
from chainer.functions import sign  # NOQA
from chainer.functions import sqrt  # NOQA
from chainer.functions import square  # NOQA

# Padding
from chainer.functions import pad  # NOQA

# Sorting, searching, and counting
from chainer.functions import argmax  # NOQA
from chainer.functions import argmin  # NOQA
from chainer.functions import where  # NOQA

# Statistics
from chainer.functions import max  # NOQA
from chainer.functions import max as amax  # NOQA
from chainer.functions import min  # NOQA
from chainer.functions import min as amin  # NOQA

from chainer.functions import average  # NOQA
from chainer.functions import mean  # NOQA

# Testing
from chainer.np import testing  # NOQA

# =============================================================================
# Chainer-specific APIs
# =============================================================================

# Device
from chainer.np.device import Device  # NOQA
from chainer.np.device import get_default_device  # NOQA
from chainer.np.device import get_device  # NOQA
from chainer.np.device import set_default_device  # NOQA
