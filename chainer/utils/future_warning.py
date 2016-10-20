import functools
import warnings

import chainer


def future_warning(version):

    def _future_warning(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if not chainer.enable_deprecated:
                warnings.warn('Deprecated', FutureWarning)
            return f(*args, **kwargs)
        return wrapper

    return _future_warning
