import functools
import traceback

import numpy

import chainer
from chainer import np


def _call_func(self, impl, args, kw):
    try:
        return impl(self, *args, **kw), None, None
    except Exception as e:
        return None, e, traceback.format_exc()


def _check_chainer_numpy_error(self, chainer_error, chainer_tb, numpy_error,
                               numpy_tb, accept_error):
    # TODO(oktua): expected_regexp like numpy.testing.assert_raises_regex
    if chainer_error is None and numpy_error is None:
        self.fail('Both chainer.np and numpy are expected to raise errors, '
                  'but not')
    elif chainer_error is None:
        self.fail('Only numpy raises error\n\n' + numpy_tb)
    elif numpy_error is None:
        self.fail('Only chainer.np raises error\n\n' + chainer_tb)
    elif not isinstance(chainer_error, type(numpy_error)):
        # Chainer errors should be at least as explicit as the NumPy errors,
        # i.e. allow Chainer errors to derive from NumPy errors but not the
        # opposite. This ensures that try/except blocks that catch NumPy errors
        # also catch Chainer errors.
        msg = '''Different types of errors occurred

chainer.np
%s
numpy
%s
''' % (chainer_tb, numpy_tb)
        self.fail(msg)
    elif not (isinstance(chainer_error, accept_error) and
              isinstance(numpy_error, accept_error)):
        msg = '''Both cupy and numpy raise exceptions

chainer.np
%s
numpy
%s
''' % (chainer_tb, numpy_tb)
        self.fail(msg)


def _make_decorator(check_func, name, type_check, accept_error):
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            kw[name] = np
            with np.get_device('native'):
                ch_result, ch_error, ch_tb = _call_func(self, impl, args, kw)

            # TODO(beam2d): Test CUDA device, too.

            kw[name] = numpy
            np_result, np_error, np_tb = _call_func(self, impl, args, kw)

            if ch_error or np_error:
                _check_chainer_numpy_error(self, ch_error, ch_tb,
                                           np_error, np_tb,
                                           accept_error=accept_error)
                return

            self.assertIsNotNone(ch_result)
            self.assertIsNotNone(np_result)

            self.assertEqual(ch_result.shape, np_result.shape)

            if type_check:
                self.assertEqual(ch_result.dtype, np_result.dtype,
                                 'cupy dtype is not equal to numpy dtype')
            self.assertIsInstance(ch_result, chainer.Variable)
            check_func(ch_result, np_result)

        return test_func
    return decorator


def numpy_chainer_array_equal(err_msg='', verbose=True, name='np',
                              type_check=True, accept_error=()):
    def check_func(ch_result, np_result):
        numpy.testing.assert_array_equal(ch_result.array, np_result, err_msg,
                                         verbose)

    return _make_decorator(check_func, name, type_check, accept_error)
