import unittest

import mock
import numpy

import chainer
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
    {'converter_style': 'new'})
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
        if self.converter_style == 'new':
            @chainer.dataset.converter()
            def wrapped_converter(*args, **kwargs):
                return converter_func(*args, **kwargs)

            return wrapped_converter
        assert False

    def check_converter_received_device_arg(
            self, received_device_arg, device_arg):

        new_style = self.converter_style == 'new'

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
