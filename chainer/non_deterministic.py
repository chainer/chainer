import functools
import warnings
from chainer import configuration


def non_deterministic(fn):
    """Decorator to annontate non-deterministic functions

    Use this decorator for functions that introduce non-deterministic
    functions, such as atomicAdd.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if configuration.config.check_deterministic:
            warnings.warn(
                'Non deterministic function of called while'
                ' config.check_deterministic set.')
        return fn(*args, **kwargs)
    return wrapper
