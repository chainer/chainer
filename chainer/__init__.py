from __future__ import absolute_import
import collections
import os
import threading
import warnings as builtin_warnings

import numpy

from chainer import _version
from chainer import backends  # NOQA
from chainer import dataset  # NOQA
from chainer import datasets  # NOQA
from chainer import distributions  # NOQA
from chainer import function_hooks  # NOQA
from chainer import functions  # NOQA
from chainer import graph_optimizations  # NOQA
from chainer import initializers  # NOQA
from chainer import iterators  # NOQA
from chainer import links  # NOQA
from chainer import optimizers  # NOQA
from chainer import serializers  # NOQA
from chainer import training  # NOQA
from chainer import variable  # NOQA
from chainer import warnings  # NOQA


# import class and function
# These functions from backends.cuda are kept for backward compatibility
from chainer._runtime_info import print_runtime_info  # NOQA
from chainer.backend import get_device  # NOQA
from chainer.backend import using_device  # NOQA
from chainer.backends.cuda import should_use_cudnn  # NOQA
from chainer.backends.cuda import should_use_cudnn_tensor_core  # NOQA
from chainer.configuration import config  # NOQA
from chainer.configuration import global_config  # NOQA
from chainer.configuration import using_config  # NOQA
from chainer.device_resident import DeviceResident  # NOQA
from chainer.distribution import cross_entropy  # NOQA
from chainer.distribution import Distribution  # NOQA
from chainer.distribution import kl_divergence  # NOQA
from chainer.distribution import register_kl  # NOQA
from chainer.function import force_backprop_mode  # NOQA
from chainer.function import Function  # NOQA
from chainer.function import FunctionAdapter  # NOQA
from chainer.function import no_backprop_mode  # NOQA
from chainer.function_hook import FunctionHook  # NOQA
from chainer.function_node import FunctionNode  # NOQA
from chainer.function_node import grad  # NOQA
from chainer.functions import array  # NOQA
from chainer.functions.math import basic_math  # NOQA
from chainer.graph_optimizations.static_graph import static_graph  # NOQA
from chainer.graph_optimizations.static_graph_utilities import static_code  # NOQA
from chainer.initializer import Initializer  # NOQA
from chainer.link import Chain  # NOQA
from chainer.link import ChainList  # NOQA
from chainer.link import Link  # NOQA
from chainer.link_hook import LinkHook  # NOQA
from chainer.optimizer import GradientMethod  # NOQA
from chainer.optimizer import Optimizer  # NOQA
from chainer.optimizer import UpdateRule  # NOQA
from chainer.reporter import DictSummary  # NOQA
from chainer.reporter import get_current_reporter  # NOQA
from chainer.reporter import report  # NOQA
from chainer.reporter import report_scope  # NOQA
from chainer.reporter import Reporter  # NOQA
from chainer.reporter import Summary  # NOQA
from chainer.sequential import Sequential  # NOQA
from chainer.serializer import AbstractSerializer  # NOQA
from chainer.serializer import Deserializer  # NOQA
from chainer.serializer import Serializer  # NOQA
from chainer.variable import as_variable  # NOQA
from chainer.variable import Parameter  # NOQA
from chainer.variable import Variable  # NOQA


# Alias for backward compatibility
from chainer import cuda  # NOQA


from chainer import _environment_check


import chainerx


# Introduce an alias that cannot be declared at the original place due to
# circular imports.
import chainer.utils.walker_alias
chainer.utils.WalkerAlias = chainer.utils.walker_alias.WalkerAlias
del chainer


# Check environment conditions
_environment_check.check()


__version__ = _version.__version__

_thread_local = threading.local()
_array_types = None
_cpu_array_types = None


# Used in chainer.FunctionNode.forward_chainerx().
# This value is returned to indicate that the function does not support forward
# computation in ChainerX implementation with given input arrays and other
# arguments.
class _FallbackType(object):
    def __repr__(self):
        return 'Fallback'


Fallback = _FallbackType()


def get_function_hooks():
    try:
        ret = _thread_local.function_hooks
    except AttributeError:
        ret = collections.OrderedDict()
        _thread_local.function_hooks = ret
    return ret


def _get_link_hooks():
    try:
        ret = _thread_local.link_hooks
    except AttributeError:
        ret = collections.OrderedDict()
        _thread_local.link_hooks = ret
    return ret


def _load_array_types():
    # Note: this function may not be protected by GIL because of external
    # calls.
    global _array_types
    global _cpu_array_types
    if _array_types is None:
        array_types = [numpy.ndarray]
        cpu_array_types = [numpy.ndarray]

        if backends.cuda.available:
            array_types.append(backends.cuda.ndarray)

        if backends.intel64.is_ideep_available():
            array_types.append(backends.intel64.mdarray)
            cpu_array_types.append(backends.intel64.mdarray)

        if chainerx.is_available():
            array_types.append(chainerx.ndarray)
            cpu_array_types.append(chainerx.ndarray)

        array_types = tuple(array_types)
        cpu_array_types = tuple(cpu_array_types)

        _array_types = array_types
        _cpu_array_types = cpu_array_types


def get_array_types():
    _load_array_types()
    return _array_types


def get_cpu_array_types():
    _load_array_types()
    return _cpu_array_types


# TODO(hvy): Move this function to backend?
def is_arrays_compatible(arrays):
    # Do not use this function to check if a single object is an array or
    # not. Use isinstance(obj, chainer.get_array_types()) instead.

    arrays = [a for a in arrays if a is not None]

    if not arrays:
        return True

    # If there's at least one chainerx.ndarray, all other arrays
    # will be converted to memory-shared chainerx.ndarrays.
    # TODO(niboshi): intel64.mdarray is not supported yet.
    # TODO(niboshi): Delegate array compatibility check to chainerx.
    if (chainerx.is_available()
            and any([isinstance(arr, chainerx.ndarray) for arr in arrays])):
        return not any([
            isinstance(arr, backends.intel64.mdarray) for arr in arrays])

    if isinstance(arrays[0], backends.cuda.ndarray):
        types = backends.cuda.ndarray
    else:
        types = get_cpu_array_types()
    return all([isinstance(a, types) for a in arrays])


class _Mixed16(object):

    dtype = numpy.dtype(numpy.float16)

    def __repr__(self):
        return "dtype('mixed16')"


mixed16 = _Mixed16()
"""Dtype-like object that represents 16/32 bits mixed precision float."""


global_config.debug = bool(int(os.environ.get('CHAINER_DEBUG', '0')))
global_config.cudnn_deterministic = False
global_config.warn_nondeterministic = False
global_config.enable_backprop = True
global_config.keep_graph_on_report = bool(int(
    os.environ.get('CHAINER_KEEP_GRAPH_ON_REPORT', '0')))
global_config.train = True
global_config.type_check = bool(int(os.environ.get('CHAINER_TYPE_CHECK', '1')))
global_config.use_cudnn = os.environ.get('CHAINER_USE_CUDNN', 'auto')
global_config.use_cudnn_tensor_core = 'auto'
global_config.autotune = False
global_config.schedule_func = None
global_config.use_ideep = os.environ.get('CHAINER_USE_IDEEP', 'never')
global_config.lazy_grad_sum = bool(int(
    os.environ.get('CHAINER_LAZY_GRAD_SUM', '0')))
global_config.cudnn_fast_batch_normalization = bool(int(
    os.environ.get('CHAINER_CUDNN_FAST_BATCH_NORMALIZATION', '0')))

_chainer_dtype = os.environ.get('CHAINER_DTYPE', 'float32')
if _chainer_dtype in ('float16', 'float32', 'float64'):
    global_config.dtype = numpy.dtype(_chainer_dtype)
elif _chainer_dtype == 'mixed16':
    global_config.dtype = mixed16
else:
    raise TypeError('incorrect dtype name in CHAINER_DTYPE: "{}". '
                    'Only float16/32/64 are allowed.'.format(_chainer_dtype))
global_config.in_recomputing = False


def is_debug():
    """Returns if the debug mode is enabled or not in the current thread.

    Returns:
        bool:  ``True`` if the debug mode is enabled.
    """
    return bool(config.__getattr__('debug'))


def set_debug(debug):
    """Enables or disables the debug mode in the current thread.

    .. note::

        ``chainer.set_debug(value)`` is equivalent to
        ``chainer.config.debug = value``.

    Args:
        debug (bool): New debug mode.
    """
    config.debug = debug


class DebugMode(object):
    """Debug mode context.

    This class provides a context manager for debug mode. When entering the
    context, it sets the debug mode to the value of `debug` parameter with
    memorizing its original value. When exiting the context, it sets the debug
    mode back to the original value.

    .. deprecated:: v2.0.0

        Use :func:`chainer.using_config` instead. See :ref:`debug` for details.

    Args:
        debug (bool): Debug mode used in the context.
    """

    def __init__(self, debug):
        builtin_warnings.warn(
            'chainer.DebugMode is deprecated. '
            'Use chainer.using_config("debug", ...) instead.',
            DeprecationWarning)
        self._using = using_config('debug', debug)

    def __enter__(self):
        self._using.__enter__()

    def __exit__(self, *args):
        self._using.__exit__(*args)


def get_dtype(dtype=None, map_mixed16=None):
    """Resolves Chainer's default dtype.

    Args:
        dtype: Dtype specifier. If this value is specified (not ``None``),
            this function returns the dtype object corresponding to it.
        map_mixed16: Dtype specifier. When ``chainer.config.dtype`` is mixed16,
            this option is used. If this value is ``None``, float16 is used.

    Returns:
        If ``dtype`` is not ``None``, it returns the dtype normalized by
        ``numpy.dtype()``. Otherwise, it returns ``chainer.config.dtype`` (see
        :ref:`configuration`) normalized as well. When ``chainer.config.dtype``
        is :data:`~chainer.mixed16` and ``map_mixed16`` is specified, it
        returns the normalized version of ``map_mixed16``.

    """
    if dtype is None:
        dtype = config.dtype
    if dtype is mixed16 and map_mixed16 is not None:
        dtype = map_mixed16
    return numpy.dtype(dtype)


basic_math.install_variable_arithmetics()
array.get_item.install_variable_get_item()

disable_experimental_feature_warning = False
