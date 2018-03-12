from functools import wraps
import inspect
import os
import warnings

import chainer
import cupy
import numpy

from benchmarks.utils.helper import _is_func
from benchmarks.utils.helper import parameterize
from benchmarks.utils.helper import sync


_backend_modes = [
    # GPU (with use_cudnn == 'never')
    'gpu',

    # GPU (with use_cudnn == 'auto')
    'gpu-cudnn',

    # CPU (with use_ideep == 'never')
    'cpu',

    # CPU (with use_ideep == 'auto')
    'cpu-ideep',
]


def backends(*modes):
    """Class decorator to parameterize the benchmark class with backends.

    This is a special form of :func:`parameterize` to parameterize the
    backend variation. For all `time_*` functions and `setup` function
    in the class, this decorator:

    * wraps the function to be called with the Chainer configuration
      (`use_cudnn` and `use_ideep`) set to the current backend variation.
    * wraps the function to perform CPU/GPU synchronization after the
      benchmark, when the current backend variation uses GPU. The time
      taken for synchronization is counted as a elapsed time in the benchmark.
    * injects the array module (`cupy` or `numpy` depending on the current
      variation) as `self.xp` so that benchmark code can use it to work with
      array modules with each backend.
    * provides access to `is_backend_gpu()` and `is_backend_ideep()` methods
      so that benchmark code can use it to change behavior depending on the
      backend variation (e.g., `if is_backend_gpu(): model.to_gpu()`).

    This decorator adds parameter axis with the name of `backend`.

    Note that `cpu-ideep` mode will automatically be skipped if the current
    benchmark setup does not support it, e.g., when running benchmark
    against older Chainer version that does not support iDeep.

    You cannot apply `parameterize` decorator to the class already decorated
    by this decorator.  If you want to use `parameterize` along with this
    decorator, make `parameterize` the most inner (i.e., the closest to the
    class declaration) decorator.

    Example of usage is as follows:

    >>> @backend('gpu', 'gpu-cudnn', 'cpu', 'cpu-ideep')
    ... class ConvolutionBenchmark(object):
    ...     def time_benchmark(self):
    ...         ...
    """

    assert all([m in _backend_modes for m in modes])

    def _wrap_class(klass):
        assert isinstance(klass, type)
        return _inject_backend_mode(klass, modes)

    return _wrap_class


def _inject_backend_mode(klass, modes):
    klass = parameterize([('backend', modes)])(klass)

    # `setup` method is mandatory to inject backends to skip axis.
    if not hasattr(klass, 'setup'):
        def _setup(self, *args, **kwargs):
            pass
        klass.setup = _setup

    members = inspect.getmembers(klass, predicate=_is_func)

    for (name, func) in members:
        if not (name == 'setup' or name.startswith('time_')):
            continue

        def _wrap_func(f):
            @wraps(f)
            def _wrapped_func(self, backend, *args, **kwargs):
                _benchmark_backend_gpu = False
                _benchmark_backend_ideep = False
                xp = numpy
                use_cudnn = 'never'
                use_ideep = 'never'

                target = f
                if backend.startswith('gpu'):
                    xp = cupy
                    _benchmark_backend_gpu = True
                    target = sync(target)
                    if 'cudnn' in backend:
                        use_cudnn = 'auto'
                elif 'ideep' in backend:
                    if not have_ideep():
                        # Raise in `setup` to skip this parameter axis.
                        warnings.warn('iDeep is unavailable')
                        raise NotImplementedError
                    use_ideep = 'auto'
                    _benchmark_backend_ideep = True

                with _BackendConfig({
                        'use_cudnn': use_cudnn,
                        'use_ideep': use_ideep,
                        '_benchmark_backend_gpu': _benchmark_backend_gpu,
                        '_benchmark_backend_ideep': _benchmark_backend_ideep,
                        }):

                    # Inject self.xp
                    assert not hasattr(self, 'xp')
                    setattr(self, 'xp', xp)
                    target(self, *args, **kwargs)
                    delattr(self, 'xp')

            return _wrapped_func
        setattr(klass, name, _wrap_func(func))

    return klass


class _BackendConfig(object):
    """Context manager that changes multiple Chainer configurations."""

    def __init__(self, params):
        self._params = params
        self._contexts = []

    def __enter__(self):
        self._contexts = [
            chainer.using_config(k, v) for (k, v) in self._params.items()
        ]
        for c in self._contexts:
            c.__enter__()
        return self

    def __exit__(self, typ, value, traceback):
        for c in reversed(self._contexts):
            c.__exit__(typ, value, traceback)


def is_backend_gpu():
    """Returns True if the current backend is GPU."""

    return chainer.config._benchmark_backend_gpu


def is_backend_ideep():
    """Returns True if the current backend is iDeep."""

    return chainer.config._benchmark_backend_ideep


def have_ideep():
    """Tests if iDeep can be used in the current benchmark configuration.

    If you intend to write benchmark for iDeep outside of `backend` decorator,
    first make sure that iDeep is available using this function.
    This makes possible to run the same benchmark code over past versions of
    Chainer (prior to iDeep support).
    """

    try:
        import chainer.backends.intel64
    except ImportError:
        return False
    return chainer.backends.intel64.is_ideep_available()
