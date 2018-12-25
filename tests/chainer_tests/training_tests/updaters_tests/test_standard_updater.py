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


class TestUpdater(unittest.TestCase):

    def setUp(self):
        self.target = chainer.Link()
        self.iterator = DummyIterator([(numpy.array(1), numpy.array(2))])
        self.optimizer = DummyOptimizer()
        self.optimizer.setup(self.target)
        self.updater = training.updaters.StandardUpdater(
            self.iterator, self.optimizer)

    def test_init_values(self):
        self.assertIsNone(self.updater.device)
        self.assertIsNone(self.updater.loss_func)
        self.assertEqual(self.updater.iteration, 0)

    def test_epoch(self):
        self.assertEqual(self.updater.epoch, 1)

    def test_new_epoch(self):
        self.assertTrue(self.updater.is_new_epoch)

    def test_get_iterator(self):
        self.assertIs(self.updater.get_iterator('main'), self.iterator)

    def test_get_optimizer(self):
        self.assertIs(self.updater.get_optimizer('main'), self.optimizer)

    def test_get_all_optimizers(self):
        self.assertEqual(self.updater.get_all_optimizers(),
                         {'main': self.optimizer})

    def test_update(self):
        self.updater.update()
        self.assertEqual(self.updater.iteration, 1)
        self.assertEqual(self.optimizer.epoch, 1)
        self.assertEqual(self.iterator.next_called, 1)

    def test_use_auto_new_epoch(self):
        self.assertTrue(self.optimizer.use_auto_new_epoch)

    def test_finalizer(self):
        self.updater.finalize()
        self.assertEqual(self.iterator.finalize_called, 1)

    def test_serialize(self):
        serializer = DummySerializer()
        self.updater.serialize(serializer)

        self.assertEqual(len(self.iterator.serialize_called), 1)
        self.assertEqual(self.iterator.serialize_called[0].path,
                         ['iterator:main'])

        self.assertEqual(len(self.optimizer.serialize_called), 1)
        self.assertEqual(
            self.optimizer.serialize_called[0].path, ['optimizer:main'])

        self.assertEqual(serializer.called, [('iteration', 0)])


class TestUpdaterUpdateArguments(unittest.TestCase):

    def setUp(self):
        self.target = chainer.Link()
        self.optimizer = DummyOptimizer()
        self.optimizer.setup(self.target)

    def test_update_tuple(self):
        iterator = DummyIterator([(numpy.array(1), numpy.array(2))])
        updater = training.updaters.StandardUpdater(iterator, self.optimizer)

        updater.update_core()

        self.assertEqual(self.optimizer.update.call_count, 1)
        args, kwargs = self.optimizer.update.call_args
        self.assertEqual(len(args), 3)
        loss, v1, v2 = args
        self.assertEqual(len(kwargs), 0)

        self.assertIs(loss, self.optimizer.target)
        self.assertIsInstance(v1, numpy.ndarray)
        self.assertEqual(v1, 1)
        self.assertIsInstance(v2, numpy.ndarray)
        self.assertEqual(v2, 2)

        self.assertEqual(iterator.next_called, 1)

    def test_update_dict(self):
        iterator = DummyIterator([{'x': numpy.array(1), 'y': numpy.array(2)}])
        updater = training.updaters.StandardUpdater(iterator, self.optimizer)

        updater.update_core()

        self.assertEqual(self.optimizer.update.call_count, 1)
        args, kwargs = self.optimizer.update.call_args
        self.assertEqual(len(args), 1)
        loss, = args
        self.assertEqual(set(kwargs.keys()), {'x', 'y'})

        v1 = kwargs['x']
        v2 = kwargs['y']
        self.assertIs(loss, self.optimizer.target)
        self.assertIsInstance(v1, numpy.ndarray)
        self.assertEqual(v1, 1)
        self.assertIsInstance(v2, numpy.ndarray)
        self.assertEqual(v2, 2)

        self.assertEqual(iterator.next_called, 1)

    def test_update_var(self):
        iterator = DummyIterator([numpy.array(1)])
        updater = training.updaters.StandardUpdater(iterator, self.optimizer)

        updater.update_core()

        self.assertEqual(self.optimizer.update.call_count, 1)
        args, kwargs = self.optimizer.update.call_args
        self.assertEqual(len(args), 2)
        loss, v1 = args
        self.assertEqual(len(kwargs), 0)

        self.assertIs(loss, self.optimizer.target)
        self.assertIsInstance(v1, numpy.ndarray)
        self.assertEqual(v1, 1)

        self.assertEqual(iterator.next_called, 1)


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
class TestUpdaterCustomConverter(unittest.TestCase):

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

    def check_converter_in_arrays(self, device_arg):
        iterator = DummyIterator([(numpy.array(1), numpy.array(2))])
        optimizer = self.create_optimizer()

        called = [0]

        def converter(batch, device):
            if device_arg is None:
                self.assertIs(device, None)
            elif isinstance(device_arg, int):
                self.assertEqual(device, device_arg)
            elif isinstance(device_arg, _cpu.CpuDevice):
                self.assertIsInstance(device, int)
                self.assertEqual(device, -1)
            elif isinstance(device_arg, cuda.GpuDevice):
                self.assertIsInstance(device, int)
                self.assertEqual(device, device_arg.device.id)
            else:
                assert False

            self.assertIsInstance(batch, list)
            self.assertEqual(len(batch), 1)
            samples = batch[0]
            self.assertIsInstance(samples, tuple)
            self.assertEqual(len(samples), 2)
            self.assertIsInstance(samples[0], numpy.ndarray)
            self.assertIsInstance(samples[1], numpy.ndarray)
            self.assertEqual(samples[0], 1)
            self.assertEqual(samples[1], 2)
            called[0] += 1
            return samples

        updater = self.create_updater(
            iterator, optimizer, converter, device_arg)
        updater.update_core()
        self.assertEqual(called[0], 1)

    def check_converter_in_obj(self, device_arg):
        obj1 = object()
        obj2 = object()
        iterator = DummyIterator([obj1, obj2])
        optimizer = self.create_optimizer()

        called = [0]

        def converter(batch, device):
            if device_arg is None:
                self.assertIs(device, None)
            elif isinstance(device_arg, int):
                self.assertEqual(device, device_arg)
            elif isinstance(device_arg, _cpu.CpuDevice):
                self.assertIsInstance(device, int)
                self.assertEqual(device, -1)
            elif isinstance(device_arg, cuda.GpuDevice):
                self.assertIsInstance(device, int)
                self.assertEqual(device, device_arg.device.id)
            else:
                assert False

            self.assertIsInstance(batch, list)
            self.assertEqual(len(batch), 2)
            self.assertIs(batch[0], obj1)
            self.assertIs(batch[1], obj2)
            called[0] += 1
            return obj1, obj2

        updater = self.create_updater(
            iterator, optimizer, converter, device_arg)
        updater.update_core()
        self.assertEqual(called[0], 1)

    def check_converter_out_tuple(self, device_arg):
        iterator = DummyIterator([object()])
        optimizer = self.create_optimizer()
        converter_out = (object(), object())

        def converter(batch, device):
            return converter_out

        updater = self.create_updater(
            iterator, optimizer, converter, device_arg)
        updater.update_core()

        self.assertEqual(optimizer.update.call_count, 1)
        args, kwargs = optimizer.update.call_args
        self.assertEqual(len(args), 3)
        loss, v1, v2 = args
        self.assertEqual(len(kwargs), 0)

        self.assertIs(loss, optimizer.target)
        self.assertIs(v1, converter_out[0])
        self.assertIs(v2, converter_out[1])

    def check_converter_out_dict(self, device_arg):
        iterator = DummyIterator([object()])
        optimizer = self.create_optimizer()
        converter_out = {'x': object(), 'y': object()}

        def converter(batch, device):
            return converter_out

        updater = self.create_updater(
            iterator, optimizer, converter, device_arg)
        updater.update_core()

        self.assertEqual(optimizer.update.call_count, 1)
        args, kwargs = optimizer.update.call_args
        self.assertEqual(len(args), 1)
        loss, = args
        self.assertEqual(len(kwargs), 2)

        self.assertIs(loss, optimizer.target)
        self.assertEqual(sorted(kwargs.keys()), ['x', 'y'])
        self.assertIs(kwargs['x'], converter_out['x'])
        self.assertIs(kwargs['y'], converter_out['y'])

    def check_converter_out_obj(self, device_arg):
        iterator = DummyIterator([object()])
        optimizer = self.create_optimizer()
        converter_out = object()

        def converter(batch, device):
            return converter_out

        updater = self.create_updater(
            iterator, optimizer, converter, device_arg)
        updater.update_core()

        self.assertEqual(optimizer.update.call_count, 1)
        args, kwargs = optimizer.update.call_args
        self.assertEqual(len(args), 2)
        loss, v1 = args
        self.assertEqual(len(kwargs), 0)

        self.assertIs(loss, optimizer.target)
        self.assertIs(v1, converter_out)


testing.run_module(__name__, __file__)
