from functools import wraps
import inspect

import cupy


def _is_func(target):
    return inspect.ismethod(target) or inspect.isfunction(target)


def sync(target):
    """Decorator to perform CPU/GPU synchronization.

    This decorator can be applied to both classes and functions.
    """

    if isinstance(target, type):
        klass = target
        members = inspect.getmembers(klass, predicate=_is_func)
        for (name, func) in members:
            if not (name == 'setup' or name.startswith('time_')):
                continue
            setattr(klass, name, _synchronized_func(func))
        return klass
    elif _is_func(target):
        return _synchronized_func(target)
    else:
        raise TypeError('cannot apply decorator to {}'.format(target))


def _synchronized_func(func):
    @wraps(func)
    def _wrap_func(*args, **kwargs):
        event = cupy.cuda.stream.Event()
        event.record()
        event.synchronize()
        func(*args, **kwargs)
        event = cupy.cuda.stream.Event()
        event.record()
        event.synchronize()
    return _wrap_func


def parameterize(args):
    """Class decorator to parameterize the benchmark.

    Pass the list of pair of parameter name and values. Each parameter
    value will be passed as the function argument when benchmark runs.
    See the example below for the usage.

    >>> @parameterize([
    ...     ('batchsize', [32, 64, 128]),
    ...     ('n_gpus', [1, 2]),
    ... ])
    ... class MyBenchmark(object):
    ...     def time_all(self, batchsize, n_gpus):
    ...         ...

    Parameters cannot be sparse due to the limitation of ASV.
    """

    def _wrap_class(klass):
        """Wraps the given class.

        Internally, this function utilizes the parameterization feature of
        ASV, i.e., set `params` and `param_names` attribute of the class.
        `params` is a list of list of parameters, and `param_names` is a list
        of parameter names. `params[i]` is a list of parameters for parameter
        named `param_names[i]` where `i` is an index.
        """

        assert isinstance(klass, type)

        params = [arg[1] for arg in args]
        param_names = [arg[0] for arg in args]

        orig_params = getattr(klass, 'params', [])
        orig_param_names = getattr(klass, 'param_names', [])

        if 0 < len(orig_params):
            # ASV allows specifying list of parameters (instead of list of
            # list of parameters) if only one parameter axis is given.
            if not isinstance(orig_params[0], (tuple, list)):
                orig_params = [orig_params]
                if len(orig_param_names) == 0:
                    orig_param_names = ['param']
                assert len(orig_param_names) == 1
        else:
            assert len(orig_param_names) == 0

        params += orig_params
        param_names += orig_param_names

        assert len(params) == len(param_names)

        setattr(klass, 'params', params)
        setattr(klass, 'param_names', param_names)

        return klass

    return _wrap_class
