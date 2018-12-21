import functools
import itertools
import sys
import types
import unittest

import six

from chainer.testing import _gen


def _parameterize_test_case_generator(base, params):
    # Defines the logic to generate parameterized test case classes.

    for i, param in enumerate(params):
        cls_name = '%s_param_%d' % (base.__name__, i)

        def __str__(self):
            name = base.__str__(self)
            return '%s  parameter: %s' % (name, param)

        mb = {'__str__': __str__}
        for k, v in six.iteritems(param):
            if isinstance(v, types.FunctionType):

                def create_new_v():
                    f = v

                    def new_v(self, *args, **kwargs):
                        return f(*args, **kwargs)
                    return new_v

                mb[k] = create_new_v()
            else:
                mb[k] = v

        def method_generator(base_method):
            # Generates a wrapped test method

            @functools.wraps(base_method)
            def new_method(self, *args, **kwargs):
                try:
                    return base_method(self, *args, **kwargs)
                except unittest.SkipTest:
                    raise
                except Exception as e:
                    s = six.StringIO()
                    s.write('Parameterized test failed.\n\n')
                    s.write('Base test method: {}.{}\n'.format(
                        base.__name__, base_method.__name__))
                    s.write('Test parameters:\n')
                    for k, v in six.iteritems(param):
                        s.write('  {}: {}\n'.format(k, v))
                    s.write('\n')
                    s.write('{}: {}\n'.format(type(e).__name__, e))
                    e_new = AssertionError(s.getvalue())
                    if sys.version_info < (3,):
                        six.reraise(AssertionError, e_new, sys.exc_info()[2])
                    else:
                        six.raise_from(
                            e_new.with_traceback(e.__traceback__), None)
            return new_method

        yield (cls_name, mb, method_generator)


def parameterize(*params):
    return _gen.make_decorator(
        lambda base: _parameterize_test_case_generator(base, params))


def product(parameter):
    if isinstance(parameter, dict):
        keys = sorted(parameter)
        values = [parameter[key] for key in keys]
        values_product = itertools.product(*values)
        return [dict(zip(keys, vals)) for vals in values_product]

    elif isinstance(parameter, list):
        # list of lists of dicts
        if not all(isinstance(_, list) for _ in parameter):
            raise TypeError('parameter must be list of lists of dicts')
        if not all(isinstance(_, dict) for l in parameter for _ in l):
            raise TypeError('parameter must be list of lists of dicts')

        lst = []
        for dict_lst in itertools.product(*parameter):
            a = {}
            for d in dict_lst:
                a.update(d)
            lst.append(a)
        return lst

    else:
        raise TypeError(
            'parameter must be either dict or list. Actual: {}'.format(
                type(parameter)))


def product_dict(*parameters):
    return [
        {k: v for dic in dicts for k, v in six.iteritems(dic)}
        for dicts in itertools.product(*parameters)]
