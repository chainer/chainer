import collections
import inspect
import sys
import unittest


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


def make_decorator(test_case_generator):
    # `test_case_generator` is a callable that receives the source TestCase
    # class and returns an iterable of generated test cases.
    # Each element of the iterable is a 3-element tuple:
    # [0] Generated class name
    # [1] Dict of members
    # [2] Method generator
    # The method generator is also a callable that receives an original test
    # method and returns a new test method.

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
            l = _generate_test_cases(mod_name, klass, test_case_generator)
            generated_cases += l

        # Return the bundle of generated cases to allow repeated application of
        # parameterize decorators.
        return _ParameterizedTestCaseBundle(generated_cases)
    return f


def _generate_case(base, module, cls_name, mb, method_generator):
    # Returns a _TestCaseTuple.
    # Add parameters as members

    cls = type(cls_name, (base,), mb)

    # ismethod for Python 2 and isfunction for Python 3
    members = inspect.getmembers(
        cls, predicate=lambda _: inspect.ismethod(_) or inspect.isfunction(_))
    for name, method in members:
        if name.startswith('test_'):
            setattr(cls, name, method_generator(method))

    # Add new test class to module
    setattr(module, cls_name, cls)

    return _TestCaseTuple(cls, module.__name__, cls_name)


def _generate_test_cases(module_name, base_class, test_case_generator):
    # Returns a list of _TestCaseTuple's holding generated test cases.
    module = sys.modules[module_name]

    generated_cases = []
    for cls_name, members, method_generator in (
            test_case_generator(base_class)):
        c = _generate_case(
            base_class, module, cls_name, members, method_generator)
        generated_cases.append(c)

    return generated_cases
