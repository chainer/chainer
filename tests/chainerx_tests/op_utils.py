import inspect
import sys

import numpy
import pytest

import chainer
import chainer.testing
import chainerx


class OpTest(chainer.testing.function_link.FunctionTestBase):

    """Base class for op test.

    It must be used in conjunction with `op_test` decorator.

    Examples:

    @op_utils.op_test(['native:0', 'cuda:0'])
    class test_relu(op_utils.OpTest):

        # ReLU function has a non-differentiable point around zero, so
        # dodge_nondifferentiable should be set to True.
        dodge_nondifferentiable = True

        def setup(self, float_dtype):
            self.dtype = float_dtype

        def generate_inputs(self):
            dtype = self.dtype
            x = numpy.random.uniform(-1, 1, (1, 3)).astype(dtype)
            return x, w, b

        def forward_chainerx(self, inputs):
            x, w, b = inputs
            y = chainerx.relu(x)
            return y,

        def forward_expected(self, inputs):
            x, w, b = inputs
            expected = x.copy()
            expected[expected < 0] = 0
            return expected,

    In this example, `float_dtype` is a Pytest fixture for parameterizing
    floating-point dtypes (i.e. float16, float32, float64). As seen from
    this, arguments in the `setup` method are treated as Pytest fixtures.

    Test implementations must at least override the following methods:
      * `generate_inputs`: Generates inputs to the test target.
      * `forward_chainerx`: Forward implementation using ChainerX.
      * `forward_expected`: Forward reference implementation.

    It can have the same attributes as `chainer.testing.FunctionTestCase`.
    """

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


class ChainerOpTest(OpTest):

    """Base class for op test that compares the output with Chainer
    implementation.

    It must be used in conjunction with `op_test` decorator.

    Examples:

    @op_utils.op_test(['native:0', 'cuda:0'])
    class test_conv(op_utils.ChainerOpTest):

        def setup(self, float_dtype):
            self.dtype = float_dtype

        def generate_inputs(self):
            dtype = self.dtype
            x = numpy.random.uniform(-1, 1, (1, 3)).astype(dtype)
            w = numpy.random.uniform(-1, 1, (5, 3)).astype(dtype)
            b = numpy.random.uniform(-1, 1, (5,)).astype(dtype)
            return x, w, b

        def forward_chainerx(self, inputs):
            x, w, b = inputs
            y = chainerx.conv(x, w, b, self.stride, self.pad, self.cover_all)
            return y,

        def forward_chainer(self, inputs):
            x, w, b = inputs
            y = chainer.functions.convolution_nd(
                x, w, b, self.stride, self.pad, self.cover_all)
            return y,

    In this example, `float_dtype` is a Pytest fixture for parameterizing
    floating-point dtypes (i.e. float16, float32, float64). As seen from
    this, arguments in the `setup` method are treated as Pytest fixtures.

    Test implementations must at least override the following methods:
      * `generate_inputs`: Generates inputs to the test target.
      * `forward_chainerx`: Forward implementation using ChainerX.
      * `forward_chainer`: Forward reference implementation using Chainer.

    It can have the same attributes as `chainer.testing.FunctionTestCase`.
    """

    def forward_expected(self, inputs):
        output_vars = self.forward_chainer(inputs)
        return tuple([y.array for y in output_vars])

    def forward_chainerx(self, inputs):
        raise NotImplementedError(
            'Op test implementation must override `forward_chainerx`.')

    def forward_chainer(self, inputs):
        raise NotImplementedError(
            'Op test implementation must override `forward_chainer`.')


class NumpyOpTest(OpTest):

    """Base class for op test that compares the output with NumPy
    implementation.

    It must be used in conjunction with `op_test` decorator.

    Examples:

    @op_utils.op_test(['native:0', 'cuda:0'])
    class test_tanh(op_utils.NumpyOpTest):

        def setup(self, float_dtype):
            self.dtype = dtype

        def generate_inputs(self):
            x = numpy.random.uniform(-1, 1, (2, 3)).astype(self.dtype)
            return x,

        def forward_xp(self, inputs, xp):
            x, = inputs
            return xp.tanh(x),

    In this example, `float_dtype` is a Pytest fixture for parameterizing
    floating-point dtypes (i.e. float16, float32, float64). As seen from
    this, arguments in the `setup` method are treated as Pytest fixtures.

    Test implementations must at least override the following methods:
      * `generate_inputs`: Generates inputs to the test target.
      * `forward_xp`: Forward implementation using both ChainerX and NumPy.

    It can have the same attributes as `chainer.testing.FunctionTestCase`.

    This test also compares strides of forward output arrays with NumPy
    outputs. Set ``check_numpy_strides_compliance`` attribute to ``False``
    to skip this check.
    """

    check_numpy_strides_compliance = True

    def forward_chainerx(self, inputs):
        return self.forward_xp(inputs, chainerx)

    def forward_expected(self, inputs):
        outputs = self.forward_xp(inputs, numpy)
        return tuple([numpy.asarray(y) for y in outputs])

    def forward_xp(self, inputs, xp):
        raise NotImplementedError(
            'Op test implementation must override `forward_xp`.')

    def check_forward_outputs(self, outputs, expected_outputs):
        super(NumpyOpTest, self).check_forward_outputs(
            outputs, expected_outputs)
        if self.check_numpy_strides_compliance:
            if not all(
                    a.strides == e.strides
                    for a, e in zip(outputs, expected_outputs)):
                msg = (
                    'Strides do not match with NumPy outputs.\n'
                    'Expected shapes and dtypes: {}\n'
                    'Actual shapes and dtypes:   {}\n'
                    'Expected strides: {}\n'
                    'Actual strides:   {}\n'.format(
                        chainer.utils._format_array_props(expected_outputs),
                        chainer.utils._format_array_props(outputs),
                        ', '.join(str(e.strides) for e in expected_outputs),
                        ', '.join(str(a.strides) for a in outputs)))
                chainer.testing.FunctionTestError.fail(msg)


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

    # We enforce 'Test' prefix in OpTest implementations so that they look like
    # unittest.TestCase implementations. OTOH generated entry function must
    # have a prefix 'test_' in order for it to be found in pytest test
    # collection.
    if not cls.__name__.startswith('Test'):
        raise TypeError(
            'OpTest class name must start with \'Test\'. Actual: {!r}'.format(
                cls.__name__))

    func_name = 'test_{}_{}'.format(cls.__name__[len('Test'):], func_suffix)

    @pytest.mark.parametrize_device(devices)
    def entry_func(device, *args, **kwargs):
        obj = cls()
        run_test_method = getattr(obj, method_name)
        try:
            obj.setup(*args, **kwargs)
            run_test_method(_make_backend_config(device.name))
        finally:
            obj.teardown()

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
    """Decorator to set up an op test.

    This decorator can be used in conjunction with either ``NumpyOpTest`` or
    ``ChainerOpTest`` to define an op test.

    See the documentation of the respective classes for detailed explanation
    and examples.

    Args:
        devices: List of devices to test.
    """

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
