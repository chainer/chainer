import unittest

import mock
import numpy
import pytest

import chainer
from chainer import backend
from chainer.backends import _cpu
from chainer.backends import cuda
from chainer import dataset
from chainer import testing
from chainer.testing import attr
from chainer import training


class DummyIterator(dataset.Iterator):

    epoch = 1
    is_new_epoch = True

    def __init__(self, next_data):
        self.finalize_called = 0
        self.next_called = 0
        self.next_data = next_data
        self.serialize_called = []

    def finalize(self):
        self.finalize_called += 1

    def __next__(self):
        self.next_called += 1
        return self.next_data

    def serialize(self, serializer):
        self.serialize_called.append(serializer)


class DummyOptimizer(chainer.Optimizer):

    def __init__(self):
        self.update = mock.MagicMock()
        self.serialize_called = []

    def serialize(self, serializer):
        self.serialize_called.append(serializer)


class DummySerializer(chainer.Serializer):

    def __init__(self, path=None):
        if path is None:
            path = []
        self.path = path
        self.called = []

    def __getitem__(self, key):
        return DummySerializer(self.path + [key])

    def __call__(self, key, value):
        self.called.append((key, value))


class TestStandardUpdater(unittest.TestCase):

    def setUp(self):
        self.target = chainer.Link()
        self.iterator = DummyIterator([(numpy.array(1), numpy.array(2))])
        self.optimizer = DummyOptimizer()
        self.optimizer.setup(self.target)
        self.updater = training.updaters.StandardUpdater(
            self.iterator, self.optimizer)

    def test_init_values(self):
        assert self.updater.device is None
        assert self.updater.input_device is None
        assert self.updater.loss_func is None
        assert self.updater.iteration == 0

    def test_epoch(self):
        assert self.updater.epoch == 1

    def test_new_epoch(self):
        assert self.updater.is_new_epoch is True

    def test_get_iterator(self):
        assert self.updater.get_iterator('main') is self.iterator

    def test_get_optimizer(self):
        assert self.updater.get_optimizer('main') is self.optimizer

    def test_get_all_optimizers(self):
        assert self.updater.get_all_optimizers() == {'main': self.optimizer}

    def test_update(self):
        self.updater.update()
        assert self.updater.iteration == 1
        assert self.optimizer.epoch == 1
        assert self.iterator.next_called == 1

    def test_use_auto_new_epoch(self):
        assert self.optimizer.use_auto_new_epoch is True

    def test_finalizer(self):
        self.updater.finalize()
        assert self.iterator.finalize_called == 1

    def test_serialize(self):
        serializer = DummySerializer()
        self.updater.serialize(serializer)

        assert len(self.iterator.serialize_called) == 1
        assert self.iterator.serialize_called[0].path == ['iterator:main']

        assert len(self.optimizer.serialize_called) == 1
        assert self.optimizer.serialize_called[0].path == ['optimizer:main']

        assert serializer.called == [('iteration', 0)]


_backend_params = [
    # NumPy
    {},
    # CuPy
    {'use_cuda': True, 'cuda_device': 0},
    {'use_cuda': True, 'cuda_device': 1},
    # ChainerX
    {'use_chainerx': True, 'chainerx_device': 'native:0'},
    {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
    {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
]


@chainer.testing.backend.inject_backend_tests(None, _backend_params)
@chainer.testing.backend.inject_backend_tests(None, _backend_params)
@chainer.testing.backend.inject_backend_tests(None, _backend_params)
class TestStandardUpdaterDevice(unittest.TestCase):

    def test_device(
            self, model_initial_backend_config, model_backend_config,
            input_backend_config):
        model_initial_device = model_initial_backend_config.device
        device = model_backend_config.device
        input_device = input_backend_config.device

        model = chainer.Link()
        model.to_device(model_initial_device)
        optimizer = DummyOptimizer()
        optimizer.setup(model)
        iterator = DummyIterator([numpy.array(1), numpy.array(2)])

        updater = training.updaters.StandardUpdater(
            iterator,
            optimizer,
            device=device,
            input_device=input_device)

        assert updater.device is device
        assert updater.input_device is input_device

        # Check the model device.
        assert model.device == device

        updater.update_core()

        assert optimizer.update.call_count == 1
        args, kwargs = optimizer.update.call_args
        assert len(args) == 2
        assert len(kwargs) == 0
        loss, v1 = args

        # Check the input device.
        assert backend.get_device_from_array(v1) == input_device


class DummyDevice(backend.Device):

    xp = numpy
    supported_array_types = (numpy.ndarray,)

    def __init__(self, index):
        self.index = index

    def __eq__(self, other):
        return isinstance(other, DummyDevice) and other.index == self.index

    # TODO(niboshi): Define name property instead (#7149).
    def __str__(self):
        return '@dummy:{}'.format(self.index)

    def send_array(self, array):
        return array.copy()


@testing.parameterize(*testing.product({
    'omit_device': [True, False],
    'omit_input_device': [True, False],
}))
class TestStandardUpdaterDeviceArgumentFallback(unittest.TestCase):
    """Tests the fallback behavior regarding device and input_device
    arguments."""

    def test_device_argument_fallback(self):
        self.check_device_argument_fallback(
            initial_model_device=DummyDevice(0),
            initial_input_device=DummyDevice(1),
            device_arg=DummyDevice(3),
            input_device_arg=DummyDevice(4))

    @attr.multi_gpu(2)
    def test_gpu_to_gpu_transfer(self):
        initial_model_device = backend.GpuDevice.from_device_id(0)
        initial_input_device = backend.GpuDevice.from_device_id(0)
        # GpuDevice is given as device/input_device arguments:
        # - model GPU-to-GPU transfer should be skipped.
        # - input GPU-to-GPU transfer should NOT be skipped.

        # device      : GPU 1
        # input_device: Other device
        self.check_device_argument_fallback(
            initial_model_device=initial_model_device,
            initial_input_device=initial_input_device,
            device_arg=backend.GpuDevice.from_device_id(1),
            input_device_arg=DummyDevice(0))

        # device      : GPU 1
        # input_device: omitted
        self.check_device_argument_fallback(
            initial_model_device=initial_model_device,
            initial_input_device=initial_input_device,
            device_arg=backend.GpuDevice.from_device_id(1),
            input_device_arg=None)

        # device      : Other device
        # input_device: GPU 1
        self.check_device_argument_fallback(
            initial_model_device=initial_model_device,
            initial_input_device=initial_input_device,
            device_arg=DummyDevice(0),
            input_device_arg=backend.GpuDevice.from_device_id(1))

        # device      : omitted
        # input_device: GPU 1
        self.check_device_argument_fallback(
            initial_model_device=initial_model_device,
            initial_input_device=initial_input_device,
            device_arg=None,
            input_device_arg=backend.GpuDevice.from_device_id(1))

    def _get_expected_devices(
            self,
            initial_model_device,
            initial_input_device,
            device_arg,
            input_device_arg):
        # Determines the expected devices.
        # Returns: (
        #    expected_device_attr:   Expected StandardUpdater.device
        #    expected_model_device:  Expected model device
        #    expected_input_device:  Expected device given to converters
        # )
        # or None if an error is expected.

        # If device_arg is given and is GpuDevice, it will skip GPU-to-GPU
        # transfer of the model (but not input).
        if (device_arg is not None
                and isinstance(device_arg, backend.GpuDevice)):
            if isinstance(initial_model_device, backend.GpuDevice):
                expected_model_device = initial_model_device
            else:
                expected_model_device = device_arg

            # If input_device is omitted, device argument should be used.
            if input_device_arg is None:
                expected_input_device = device_arg
            else:
                expected_input_device = input_device_arg

            expected_device_attr = device_arg

            return (
                expected_device_attr,
                expected_model_device,
                expected_input_device)

        # expect_table
        # Key:   (omit_device, omit_input_device)
        # Value: (expected_device, expected_input_device)
        #
        # None means unchanged.
        expect_table = {
            (0, 0): (device_arg, input_device_arg),
            (0, 1): (device_arg, device_arg),
            (1, 0): (None, input_device_arg),
            (1, 1): (None, None),
        }
        omit_device = 1 if device_arg is None else 0
        omit_input_device = 1 if input_device_arg is None else 0
        expected_model_device, expected_input_device = (
            expect_table[(
                omit_device,
                omit_input_device)])
        expected_device_attr = expected_model_device
        return (
            expected_device_attr,
            expected_model_device,
            expected_input_device)

    def check_device_argument_fallback(
            self,
            initial_model_device,
            initial_input_device,
            device_arg,
            input_device_arg):

        if self.omit_device:
            device_arg = None
        if self.omit_input_device:
            input_device_arg = None

        actual_converter_device_args = []

        @chainer.dataset.converter()
        def convert(arr, device):
            # The converter records the given device.
            actual_converter_device_args.append(device)
            if device is None:
                return arr
            return device.send(arr)

        class Model(chainer.Link):
            def __init__(self):
                chainer.Link.__init__(self)
                with self.init_scope():
                    self.p1 = chainer.Parameter()
                    self.p2 = chainer.Parameter(
                        numpy.array([1, 2], numpy.float32))

            def forward(self, x):
                return chainer.functions.identity(x)

        model = Model()
        model.to_device(initial_model_device)
        optimizer = DummyOptimizer()
        optimizer.setup(model)
        iterator = DummyIterator([
            initial_input_device.send(numpy.array(1)),
        ])

        # Make kwargs
        kwargs = {}
        if device_arg is not None:
            kwargs['device'] = device_arg
        if input_device_arg is not None:
            kwargs['input_device'] = input_device_arg

        # Calculate the expected devices
        expect = self._get_expected_devices(
            initial_model_device,
            initial_input_device,
            device_arg,
            input_device_arg)

        if expect is None:
            # Error is expected
            with pytest.raises(KeyError):
                training.updaters.StandardUpdater(
                    iterator, optimizer, convert, **kwargs)
            return

        (expected_device_attr,
         expected_model_device,
         expected_input_device) = expect

        # Create the StandardUpdater
        updater = training.updaters.StandardUpdater(
            iterator, optimizer, convert, **kwargs)

        assert updater.device == expected_device_attr
        assert updater.input_device == expected_input_device

        # Check the model device
        if expected_model_device is None:
            # Model device is unchanged
            expected_model_device = initial_model_device
        # TODO(niboshi): model.device should be expected_model_device too.
        assert model.p1.device == expected_model_device
        assert model.p2.device == expected_model_device

        # Process a batch
        updater.update_core()

        # Check the input device given to the converter
        assert len(actual_converter_device_args) == 1
        assert actual_converter_device_args[0] == expected_input_device


class TestStandardUpdaterDataTypes(unittest.TestCase):
    """Tests several data types with StandardUpdater"""

    def setUp(self):
        self.target = chainer.Link()
        self.optimizer = DummyOptimizer()
        self.optimizer.setup(self.target)

    def test_update_tuple(self):
        iterator = DummyIterator([(numpy.array(1), numpy.array(2))])
        updater = training.updaters.StandardUpdater(iterator, self.optimizer)

        updater.update_core()

        assert self.optimizer.update.call_count == 1
        args, kwargs = self.optimizer.update.call_args
        assert len(args) == 3
        loss, v1, v2 = args
        assert len(kwargs) == 0

        assert loss is self.optimizer.target
        assert isinstance(v1, numpy.ndarray)
        assert v1 == 1
        assert isinstance(v2, numpy.ndarray)
        assert v2 == 2

        assert iterator.next_called == 1

    def test_update_dict(self):
        iterator = DummyIterator([{'x': numpy.array(1), 'y': numpy.array(2)}])
        updater = training.updaters.StandardUpdater(iterator, self.optimizer)

        updater.update_core()

        assert self.optimizer.update.call_count == 1
        args, kwargs = self.optimizer.update.call_args
        assert len(args) == 1
        loss, = args
        assert set(kwargs.keys()) == {'x', 'y'}

        v1 = kwargs['x']
        v2 = kwargs['y']
        assert loss is self.optimizer.target
        assert isinstance(v1, numpy.ndarray)
        assert v1 == 1
        assert isinstance(v2, numpy.ndarray)
        assert v2 == 2

        assert iterator.next_called == 1

    def test_update_var(self):
        iterator = DummyIterator([numpy.array(1)])
        updater = training.updaters.StandardUpdater(iterator, self.optimizer)

        updater.update_core()

        assert self.optimizer.update.call_count == 1
        args, kwargs = self.optimizer.update.call_args
        assert len(args) == 2
        loss, v1 = args
        assert len(kwargs) == 0

        assert loss is self.optimizer.target
        assert isinstance(v1, numpy.ndarray)
        assert v1 == 1

        assert iterator.next_called == 1


@testing.parameterize(
    {'converter_style': 'old'},
    {'converter_style': 'decorator'},
    {'converter_style': 'class'})
@chainer.testing.backend.inject_backend_tests(
    ['test_converter_given_device'],
    [
        # NumPy
        {},
        # CuPy
        {'use_cuda': True, 'cuda_device': 0},
        {'use_cuda': True, 'cuda_device': 1},

        # Custom converter is not supported for ChainerX.
    ])
class TestStandardUpdaterCustomConverter(unittest.TestCase):
    """Tests custom converters of various specs"""

    def create_optimizer(self):
        target = chainer.Link()
        optimizer = DummyOptimizer()
        optimizer.setup(target)
        return optimizer

    def create_updater(self, iterator, optimizer, converter, device):
        return training.updaters.StandardUpdater(
            iterator, optimizer, converter=converter, device=device)

    def test_converter_given_device(self, backend_config):
        self.check_converter_all(backend_config.device)

    def test_converter_given_none(self):
        self.check_converter_all(None)

    def test_converter_given_int_negative(self):
        self.check_converter_all(-1)

    @attr.gpu
    def test_converter_given_int_positive(self):
        self.check_converter_all(9999)

    def check_converter_all(self, device):
        self.check_converter_in_arrays(device)
        self.check_converter_in_obj(device)
        self.check_converter_out_tuple(device)
        self.check_converter_out_dict(device)
        self.check_converter_out_obj(device)

    def get_converter(self, converter_func):
        if self.converter_style == 'old':
            return converter_func

        if self.converter_style == 'decorator':
            @chainer.dataset.converter()
            def wrapped_converter(*args, **kwargs):
                return converter_func(*args, **kwargs)

            return wrapped_converter

        if self.converter_style == 'class':
            class MyConverter(dataset.Converter):
                def __call__(self, *args, **kwargs):
                    return converter_func(*args, **kwargs)

            return MyConverter()

        assert False

    def test_converter_type(self):
        # Ensures that new-style converters inherit from dataset.Converter.

        def converter_impl(batch, device):
            pass

        converter = self.get_converter(converter_impl)

        if self.converter_style in ('decorator', 'class'):
            assert isinstance(converter, dataset.Converter)

    def check_converter_received_device_arg(
            self, received_device_arg, device_arg):

        new_style = self.converter_style in ('decorator', 'class')

        # None
        if device_arg is None:
            assert received_device_arg is None
            return

        # Normalize input device types
        is_cpu = False
        cuda_device_id = None
        if isinstance(device_arg, int):
            if device_arg < 0:
                is_cpu = True
            else:
                cuda_device_id = device_arg
        elif isinstance(device_arg, _cpu.CpuDevice):
            is_cpu = True
        elif isinstance(device_arg, cuda.GpuDevice):
            cuda_device_id = device_arg.device.id
        else:
            assert False

        # Check received device
        if is_cpu:
            if new_style:
                assert received_device_arg == _cpu.CpuDevice()
            else:
                assert received_device_arg == -1

        elif cuda_device_id is not None:
            if new_style:
                assert (received_device_arg
                        == cuda.GpuDevice.from_device_id(cuda_device_id))
            else:
                assert isinstance(received_device_arg, int)
                assert received_device_arg == cuda_device_id
        else:
            assert new_style
            assert received_device_arg is device_arg

    def check_converter_in_arrays(self, device_arg):
        iterator = DummyIterator([(numpy.array(1), numpy.array(2))])
        optimizer = self.create_optimizer()

        called = [0]

        def converter_impl(batch, device):
            self.check_converter_received_device_arg(device, device_arg)

            assert isinstance(batch, list)
            assert len(batch) == 1
            samples = batch[0]
            assert isinstance(samples, tuple)
            assert len(samples) == 2
            assert isinstance(samples[0], numpy.ndarray)
            assert isinstance(samples[1], numpy.ndarray)
            assert samples[0] == 1
            assert samples[1] == 2
            called[0] += 1
            return samples

        converter = self.get_converter(converter_impl)

        updater = self.create_updater(
            iterator, optimizer, converter, device_arg)
        updater.update_core()
        assert called[0] == 1

    def check_converter_in_obj(self, device_arg):
        obj1 = object()
        obj2 = object()
        iterator = DummyIterator([obj1, obj2])
        optimizer = self.create_optimizer()

        called = [0]

        def converter_impl(batch, device):
            self.check_converter_received_device_arg(device, device_arg)

            assert isinstance(batch, list)
            assert len(batch) == 2
            assert batch[0] is obj1
            assert batch[1] is obj2
            called[0] += 1
            return obj1, obj2

        converter = self.get_converter(converter_impl)

        updater = self.create_updater(
            iterator, optimizer, converter, device_arg)
        updater.update_core()
        assert called[0] == 1

    def check_converter_out_tuple(self, device_arg):
        iterator = DummyIterator([object()])
        optimizer = self.create_optimizer()
        converter_out = (object(), object())

        def converter_impl(batch, device):
            self.check_converter_received_device_arg(device, device_arg)
            return converter_out

        converter = self.get_converter(converter_impl)

        updater = self.create_updater(
            iterator, optimizer, converter, device_arg)
        updater.update_core()

        assert optimizer.update.call_count == 1
        args, kwargs = optimizer.update.call_args
        assert len(args) == 3
        loss, v1, v2 = args
        assert len(kwargs) == 0

        assert loss is optimizer.target
        assert v1 is converter_out[0]
        assert v2 is converter_out[1]

    def check_converter_out_dict(self, device_arg):
        iterator = DummyIterator([object()])
        optimizer = self.create_optimizer()
        converter_out = {'x': object(), 'y': object()}

        def converter_impl(batch, device):
            self.check_converter_received_device_arg(device, device_arg)
            return converter_out

        converter = self.get_converter(converter_impl)

        updater = self.create_updater(
            iterator, optimizer, converter, device_arg)
        updater.update_core()

        assert optimizer.update.call_count == 1
        args, kwargs = optimizer.update.call_args
        assert len(args) == 1
        loss, = args
        assert len(kwargs) == 2

        assert loss is optimizer.target
        assert sorted(kwargs.keys()) == ['x', 'y']
        assert kwargs['x'] is converter_out['x']
        assert kwargs['y'] is converter_out['y']

    def check_converter_out_obj(self, device_arg):
        iterator = DummyIterator([object()])
        optimizer = self.create_optimizer()
        converter_out = object()

        def converter_impl(batch, device):
            self.check_converter_received_device_arg(device, device_arg)
            return converter_out

        converter = self.get_converter(converter_impl)

        updater = self.create_updater(
            iterator, optimizer, converter, device_arg)
        updater.update_core()

        assert optimizer.update.call_count == 1
        args, kwargs = optimizer.update.call_args
        assert len(args) == 2
        loss, v1 = args
        assert len(kwargs) == 0

        assert loss is optimizer.target
        assert v1 is converter_out


testing.run_module(__name__, __file__)
