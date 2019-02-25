import inspect
import sys

import numpy
import pytest

import chainer
import chainer.testing
import chainerx


class _OpTest(chainer.testing.function.FunctionTestBase):

    def setup(self):
        # This method can be overridden by a concrete class with arbitrary
        # arguments.
        pass

    def teardown(self):
        pass

    def forward(self, inputs, device):
        # device is chainer.Device and it's ignored.
        # chainerx's default device is used instead.

        test_self = self

        class MyFunc(chainer.FunctionNode):
            def forward_chainerx(self, inputs):
                return test_self.forward_chainerx(inputs)

        return MyFunc().apply(inputs)

    def forward_chainerx(self, inputs):
        raise NotImplementedError(
            'Op test implementation must override `forward_chainerx`.')


class ChainerOpTest(_OpTest):

    def forward_expected(self, inputs):
        output_vars = self.forward_chainer(inputs)
        return tuple([y.array for y in output_vars])

    def forward_chainerx(self, inputs):
        raise NotImplementedError(
            'Op test implementation must override `forward_chainerx`.')

    def forward_chainer(self, inputs):
        raise NotImplementedError(
            'Op test implementation must override `forward_chainer`.')


class NumpyOpTest(_OpTest):

    def forward_chainerx(self, inputs):
        return self.forward_xp(inputs, chainerx)

    def forward_expected(self, inputs):
        outputs = self.forward_xp(inputs, numpy)
        return tuple([numpy.asarray(y) for y in outputs])

    def forward_xp(self, inputs, xp):
        raise NotImplementedError(
            'Op test implementation must override `forward_xp`.')


def _make_backend_config(device_name):
    backend_config = chainer.testing.BackendConfig({
        'use_chainerx': True,
        'chainerx_device': device_name,
    })
    return backend_config


def _create_test_entry_function(
        cls, module, devices, func_suffix, method_name):
    # Creates a test entry function from the template class, and places it in
    # the same module as the class.
    #
    # func_suffix:
    #    The suffix of the test entry function to create.
    # method_name:
    #    The name of the test method name defined in `FunctionTestBase` class.

    @pytest.mark.parametrize_device(devices)
    def entry_func(device, *args, **kwargs):
        obj = cls()
        run_test_method = getattr(obj, method_name)
        try:
            obj.setup(*args, **kwargs)
            run_test_method(_make_backend_config(device.name))
        finally:
            obj.teardown()

    func_name = '{}_{}'.format(cls.__name__, func_suffix)
    entry_func.__name__ = func_name

    # Set the signature of the entry function
    sig = inspect.signature(cls.setup)
    params = list(sig.parameters.values())
    params = params[1:]  # Remove `self` argument
    device_param = inspect.Parameter(
        'device', inspect.Parameter.POSITIONAL_OR_KEYWORD)
    params = [device_param] + params  # Prepend `device` argument
    entry_func.__signature__ = inspect.Signature(params)

    # Set the pytest mark
    try:
        pytestmark = cls.pytestmark
        entry_func.pytestmark += pytestmark
    except AttributeError:
        pass

    # Place the entry function in the module of the class
    setattr(module, func_name, entry_func)


def op_test(devices):

    def wrap(cls):
        # TODO(niboshi): Avoid using private entries in chainer.testing.
        if isinstance(
                cls, chainer.testing._bundle._ParameterizedTestCaseBundle):
            classes = [(c, m) for c, m, name in cls.cases]
        else:
            classes = [(cls, cls.__module__)]

        tests = [
            ('forward', 'run_test_forward'),
            ('backward', 'run_test_backward'),
            ('double_backward', 'run_test_double_backward'),
        ]
        for cls, mod in classes:
            for func_suffix, method_name in tests:
                _create_test_entry_function(
                    cls, sys.modules[mod], devices, func_suffix, method_name)

        # return None: no other decorator can be applied after this decorator.
        return None

    return wrap
