import collections
import os
import threading
import warnings

import numpy

from chainer import _version
from chainer import backends  # NOQA
from chainer import configuration  # NOQA
from chainer import dataset  # NOQA
from chainer import datasets  # NOQA
from chainer import function  # NOQA
from chainer import function_hook  # NOQA
from chainer import function_hooks  # NOQA
from chainer import function_node  # NOQA
from chainer import functions  # NOQA
from chainer import initializer  # NOQA
from chainer import initializers  # NOQA
from chainer import iterators  # NOQA
from chainer import link  # NOQA
from chainer import links  # NOQA
from chainer import optimizer  # NOQA
from chainer import optimizers  # NOQA
from chainer import reporter  # NOQA
from chainer import serializer  # NOQA
from chainer import serializers  # NOQA
from chainer import training  # NOQA
from chainer import variable  # NOQA


# import class and function
# These functions from backends.cuda are kept for backward compatibility
from chainer.backends.cuda import should_use_cudnn  # NOQA
from chainer.backends.cuda import should_use_cudnn_tensor_core  # NOQA
from chainer.configuration import config  # NOQA
from chainer.configuration import global_config  # NOQA
from chainer.configuration import using_config  # NOQA
from chainer.function import force_backprop_mode  # NOQA
from chainer.function import Function  # NOQA
from chainer.function import FunctionAdapter  # NOQA
from chainer.function import no_backprop_mode  # NOQA
from chainer.function_hook import FunctionHook  # NOQA
from chainer.function_node import FunctionNode  # NOQA
from chainer.function_node import grad  # NOQA
from chainer.functions import array  # NOQA
from chainer.functions.math import basic_math  # NOQA
from chainer.initializer import Initializer  # NOQA
from chainer.link import Chain  # NOQA
from chainer.link import ChainList  # NOQA
from chainer.link import Link  # NOQA
from chainer.optimizer import GradientMethod  # NOQA
from chainer.optimizer import Optimizer  # NOQA
from chainer.optimizer import UpdateRule  # NOQA
from chainer.reporter import DictSummary  # NOQA
from chainer.reporter import get_current_reporter  # NOQA
from chainer.reporter import report  # NOQA
from chainer.reporter import report_scope  # NOQA
from chainer.reporter import Reporter  # NOQA
from chainer.reporter import Summary  # NOQA
from chainer.serializer import AbstractSerializer  # NOQA
from chainer.serializer import Deserializer  # NOQA
from chainer.serializer import Serializer  # NOQA
from chainer.variable import as_variable  # NOQA
from chainer.variable import Parameter  # NOQA
from chainer.variable import Variable  # NOQA


# Alias for backward compatibility
from chainer import cuda  # NOQA


from chainer import _environment_check


# Check environment conditions
_environment_check.check()


__version__ = _version.__version__

_thread_local = threading.local()
_array_types = None
_cpu_array_types = None


def get_function_hooks():
    try:
        ret = _thread_local.function_hooks
    except AttributeError:
        ret = collections.OrderedDict()
        _thread_local.function_hooks = ret
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


def is_arrays_compatible(arrays):
    arrays = [a for a in arrays if a is not None]
    if len(arrays) == 0:
        return True
    if type(arrays[0]) is backends.cuda.ndarray:
        types = backends.cuda.ndarray
    else:
        types = get_cpu_array_types()
    return all([isinstance(a, types) for a in arrays])


global_config.debug = bool(int(os.environ.get('CHAINER_DEBUG', '0')))
global_config.cudnn_deterministic = False
global_config.enable_backprop = True
global_config.keep_graph_on_report = bool(int(
    os.environ.get('CHAINER_KEEP_GRAPH_ON_REPORT', '0')))
global_config.train = True
global_config.type_check = bool(int(os.environ.get('CHAINER_TYPE_CHECK', '1')))
global_config.use_cudnn = os.environ.get('CHAINER_USE_CUDNN', 'auto')
global_config.use_cudnn_tensor_core = 'auto'
global_config.autotune = False
global_config.use_ideep = os.environ.get('CHAINER_USE_IDEEP', 'never')
global_config.lazy_grad_sum = bool(int(
    os.environ.get('CHAINER_LAZY_GRAD_SUM', '0')))


def is_debug():
    """Get the debug mode.

    Returns:
        bool: Return ``True`` if Chainer is in debug mode.
    """
    return bool(config.debug)


def set_debug(debug):
    """Set the debug mode.

    .. note::

        This method changes the global state. When you use this method on
        multi-threading environment, it may affect other threads.

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
        warnings.warn('chainer.DebugMode is deprecated. '
                      'Use chainer.using_config("debug", ...) instead.',
                      DeprecationWarning)
        self._using = using_config('debug', debug)

    def __enter__(self):
        self._using.__enter__()

    def __exit__(self, *args):
        self._using.__exit__(*args)


basic_math.install_variable_arithmetics()
array.get_item.install_variable_get_item()

disable_experimental_feature_warning = False
