import collections
import functools
import inspect
import itertools
import sys
import types
import unittest

import six


# A tuple that represents a test case.
# For bare (non-generated) test cases, [1] and [2] are None.
# [0] Test case class
# [1] Module name in whicn the class is defined
# [2] Class name
_TestCaseTuple = collections.namedtuple(
    '_TestCaseTuple', ('klass', 'module_name', 'class_name'))


class _ParameterizedTestCaseBundle(object):
    def __init__(self, cases):
        # cases is a list of _TestCaseTuple's
        assert isinstance(cases, list)
        assert all(isinstance(tup, _TestCaseTuple) for tup in cases)

        self.cases = cases


def _gen_case(base, module, i, param):
    # Returns a _TestCaseTuple.

    cls_name = '%s_param_%d' % (base.__name__, i)

    # Add parameters as members

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

    cls = type(cls_name, (base,), mb)

    # Wrap test methods to generate useful error message

    def wrap_test_method(method):
        @functools.wraps(method)
        def wrap(*args, **kwargs):
            try:
                return method(*args, **kwargs)
            except unittest.SkipTest:
                raise
            except Exception as e:
                s = six.StringIO()
                s.write('Parameterized test failed.\n\n')
                s.write('Base test method: {}.{}\n'.format(
                    base.__name__, method.__name__))
                s.write('Test parameters:\n')
                for k, v in six.iteritems(param):
                    s.write('  {}: {}\n'.format(k, v))
                s.write('\n')
                s.write('{}: {}\n'.format(type(e).__name__, e))
                e_new = AssertionError(s.getvalue())
                if sys.version_info < (3,):
                    six.reraise(AssertionError, e_new, sys.exc_info()[2])
                else:
                    six.raise_from(e_new.with_traceback(e.__traceback__), None)
        return wrap

    # ismethod for Python 2 and isfunction for Python 3
    members = inspect.getmembers(
        cls, predicate=lambda _: inspect.ismethod(_) or inspect.isfunction(_))
    for name, method in members:
        if name.startswith('test_'):
            setattr(cls, name, wrap_test_method(method))

    # Add new test class to module
    setattr(module, cls_name, cls)

    return _TestCaseTuple(cls, module.__name__, cls_name)


def _gen_cases(name, base, params):
    # Returns a list of _TestCaseTuple's holding generated test cases.
    module = sys.modules[name]
    generated_cases = []
    for i, param in enumerate(params):
        c = _gen_case(base, module, i, param)
        generated_cases.append(c)
    return generated_cases


def parameterize(*params):
    def f(cases):
        if isinstance(cases, _ParameterizedTestCaseBundle):
            # The input is a parameterized test case.
            cases = cases.cases
        else:
            # Input is a bare test case, i.e. not one generated from another
            # parameterize.
            assert issubclass(cases, unittest.TestCase)
            cases = [_TestCaseTuple(cases, None, None)]

        generated_cases = []
        for klass, mod_name, cls_name in cases:
            assert issubclass(klass, unittest.TestCase)
            if mod_name is not None:
                # The input is a parameterized test case.
                # Remove it from its module.
                delattr(sys.modules[mod_name], cls_name)
            else:
                # The input is a bare test case
                mod_name = klass.__module__

            # Generate parameterized test cases out of the input test case.
            l = _gen_cases(mod_name, klass, params)
            generated_cases += l

        # Return the bundle of generated cases to allow repeated application of
        # parameterize decorators.
        return _ParameterizedTestCaseBundle(generated_cases)
    return f


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
