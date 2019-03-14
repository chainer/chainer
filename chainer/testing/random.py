from __future__ import absolute_import
import atexit
import functools
import numpy
import os
import random
import types
import unittest


from chainer.backends import cuda
from chainer.testing import _bundle


_old_python_random_state = None
_old_numpy_random_state = None


def _numpy_do_setup(deterministic=True):
    global _old_python_random_state
    global _old_numpy_random_state
    _old_python_random_state = random.getstate()
    _old_numpy_random_state = numpy.random.get_state()
    if not deterministic:
        numpy.random.seed()
    else:
        numpy.random.seed(100)


def _numpy_do_teardown():
    global _old_python_random_state
    global _old_numpy_random_state
    random.setstate(_old_python_random_state)
    numpy.random.set_state(_old_numpy_random_state)
    _old_python_random_state = None
    _old_numpy_random_state = None


def do_setup(deterministic=True):
    if cuda.available:
        cuda.cupy.testing.random.do_setup(deterministic)
    else:
        _numpy_do_setup(deterministic)


def do_teardown():
    if cuda.available:
        cuda.cupy.testing.random.do_teardown()
    else:
        _numpy_do_teardown()


# In some tests (which utilize condition.repeat or condition.retry),
# setUp/tearDown is nested. _setup_random() and _teardown_random() do their
# work only in the outermost setUp/tearDown pair.
_nest_count = 0


@atexit.register
def _check_teardown():
    assert _nest_count == 0, ('_setup_random() and _teardown_random() '
                              'must be called in pairs.')


def _setup_random():
    """Sets up the deterministic random states of ``numpy`` and ``cupy``.

    """
    global _nest_count
    if _nest_count == 0:
        nondeterministic = bool(int(os.environ.get(
            'CHAINER_TEST_RANDOM_NONDETERMINISTIC', '0')))
        do_setup(not nondeterministic)
    _nest_count += 1


def _teardown_random():
    """Tears down the deterministic random states set up by ``_setup_random``.

    """
    global _nest_count
    assert _nest_count > 0, '_setup_random has not been called'
    _nest_count -= 1
    if _nest_count == 0:
        do_teardown()


def generate_seed():
    assert _nest_count > 0, 'random is not set up'
    return numpy.random.randint(0xffffffff)


def fix_random():
    """Decorator that fixes random numbers in a test.

    This decorator can be applied to either a test case class or a test method.
    It should not be applied within ``condition.retry`` or
    ``condition.repeat``.
    """

    # TODO(niboshi): Prevent this decorator from being applied within
    #    condition.repeat or condition.retry decorators. That would repeat
    #    tests with the same random seeds. It's okay to apply this outside
    #    these decorators.

    def decorator(impl):
        if (isinstance(impl, types.FunctionType) and
                impl.__name__.startswith('test_')):
            # Applied to test method
            @functools.wraps(impl)
            def test_func(self, *args, **kw):
                _setup_random()
                try:
                    impl(self, *args, **kw)
                finally:
                    _teardown_random()
            return test_func

        if isinstance(impl, _bundle._ParameterizedTestCaseBundle):
            cases = impl
        elif isinstance(impl, type) and issubclass(impl, unittest.TestCase):
            tup = _bundle._TestCaseTuple(impl, None, None)
            cases = _bundle._ParameterizedTestCaseBundle([tup])
        else:
            raise ValueError('Can\'t apply fix_random to {}'.format(impl))

        for case, _, _ in cases.cases:
            # Applied to test case class
            klass = case

            def make_methods():
                # make_methods is required to bind the variables setUp_ and
                # tearDown_.

                setUp_ = klass.setUp
                tearDown_ = klass.tearDown

                @functools.wraps(setUp_)
                def setUp(self):
                    _setup_random()
                    setUp_(self)

                @functools.wraps(tearDown_)
                def tearDown(self):
                    try:
                        tearDown_(self)
                    finally:
                        _teardown_random()

                return setUp, tearDown

            klass.setUp, klass.tearDown = make_methods()

        return cases

    return decorator
