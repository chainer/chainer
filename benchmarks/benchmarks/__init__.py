import inspect

import chainer


# Ensure that CuPy and cuDNN are available.
assert chainer.cuda.available
assert chainer.cuda.cudnn_enabled


class BenchmarkBase(object):
    """Base class for all benchmarks.

    See also: http://asv.readthedocs.io/en/v0.2.1/writing_benchmarks.html
    """

    def __init__(self, *args, **kwargs):
        # Set pretty_name to ``<class>.<function_name>`` instead of the default
        # ``<module>.<class>.<function_name>``. This is because it is often too
        # verbose to display module name in result HTML.
        # This is a workaround needed until ASV 0.3 release.
        members = inspect.getmembers(
            self.__class__,
            predicate=lambda x: inspect.ismethod(x) or inspect.isfunction(x))
        for (name, func) in members:
            if hasattr(func, '__func__'):
                # For Python 2
                func = func.__func__
            if name.startswith('time_'):
                name = name[5:]
            func.pretty_name = '{}.{}'.format(type(self).__name__, name)

    def setup(self, *args, **kwargs):
        pass

    def setup_cache(self, *args, **kwargs):
        pass

    def teardown(self, *args, **kwargs):
        pass
