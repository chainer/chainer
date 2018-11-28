import sys
import unittest

import numpy

from chainer import backend
from chainer.backends import cuda
from chainer import dataset
from chainer import testing
from chainer.testing import attr
import chainer.testing.backend  # NOQA
import chainerx


@testing.backend.inject_backend_tests(
    None,
    [
        # NumPy
        {},
        # CuPy
        {'use_cuda': True, 'cuda_device': 0},
        {'use_cuda': True, 'cuda_device': 1},
        # ChainerX
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
    ])
class TestConcatExamples(unittest.TestCase):

    def get_arrays_to_concat(self, backend_config):
        return [
            backend_config.get_array(numpy.random.rand(2, 3))
            for _ in range(5)]

    def check_concat_arrays(self, arrays, device, expected_device):
        array = dataset.concat_examples(arrays, device)
        self.assertEqual(array.shape, (len(arrays),) + arrays[0].shape)

        assert backend.get_device_from_array(array) == expected_device

        np_array = backend.CpuDevice().send(array)
        for x, y in zip(np_array, arrays):
            numpy.testing.assert_array_equal(x, backend.CpuDevice().send(y))

    def test_concat_arrays(self, backend_config):
        arrays = self.get_arrays_to_concat(backend_config)
        self.check_concat_arrays(arrays, None, backend_config.device)

    @attr.gpu
    def test_concat_arrays_to_gpu(self, backend_config):
        arrays = self.get_arrays_to_concat(backend_config)
        self.check_concat_arrays(
            arrays, 0, backend.GpuDevice.from_device_id(0))

    @attr.chainerx
    def test_concat_arrays_to_chainerx(self, backend_config):
        device = chainerx.get_device('native:0')
        arrays = self.get_arrays_to_concat(backend_config)
        self.check_concat_arrays(
            arrays, device, backend.ChainerxDevice(device))

    def get_tuple_arrays_to_concat(self, backend_config):
        return [
            (backend_config.get_array(numpy.random.rand(2, 3)),
             backend_config.get_array(numpy.random.rand(3, 4)))
            for _ in range(5)]

    def check_concat_tuples(self, tuples, device, expected_device):
        arrays = dataset.concat_examples(tuples, device)
        self.assertEqual(len(arrays), len(tuples[0]))
        for i in range(len(arrays)):
            shape = (len(tuples),) + tuples[0][i].shape
            self.assertEqual(arrays[i].shape, shape)

            assert backend.get_device_from_array(arrays[i]) == expected_device

            arr = backend.CpuDevice().send(arrays[i])
            for x, y in zip(arr, tuples):
                numpy.testing.assert_array_equal(
                    x, backend.CpuDevice().send(y[i]))

    def test_concat_tuples(self, backend_config):
        tuples = self.get_tuple_arrays_to_concat(backend_config)
        self.check_concat_tuples(tuples, None, backend_config.device)

    @attr.gpu
    def test_concat_tuples_to_gpu(self, backend_config):
        tuples = self.get_tuple_arrays_to_concat(backend_config)
        self.check_concat_tuples(
            tuples, 0, backend.GpuDevice.from_device_id(0))

    @attr.chainerx
    def test_concat_tuples_to_chainerx(self, backend_config):
        device = chainerx.get_device('native:0')
        arrays = self.get_tuple_arrays_to_concat(backend_config)
        self.check_concat_tuples(
            arrays, device, backend.ChainerxDevice(device))

    def get_dict_arrays_to_concat(self, backend_config):
        return [
            {'x': backend_config.get_array(numpy.random.rand(2, 3)),
             'y': backend_config.get_array(numpy.random.rand(3, 4))}
            for _ in range(5)]

    def check_concat_dicts(self, dicts, device, expected_device):
        arrays = dataset.concat_examples(dicts, device)
        self.assertEqual(frozenset(arrays.keys()), frozenset(dicts[0].keys()))
        for key in arrays:
            shape = (len(dicts),) + dicts[0][key].shape
            self.assertEqual(arrays[key].shape, shape)
            self.assertEqual(
                backend.get_device_from_array(arrays[key]), expected_device)

            arr = backend.CpuDevice().send(arrays[key])
            for x, y in zip(arr, dicts):
                numpy.testing.assert_array_equal(
                    x, backend.CpuDevice().send(y[key]))

    def test_concat_dicts(self, backend_config):
        dicts = self.get_dict_arrays_to_concat(backend_config)
        self.check_concat_dicts(dicts, None, backend_config.device)

    @attr.gpu
    def test_concat_dicts_to_gpu(self, backend_config):
        dicts = self.get_dict_arrays_to_concat(backend_config)
        self.check_concat_dicts(
            dicts, 0, backend.GpuDevice.from_device_id(0))

    @attr.chainerx
    def test_concat_dicts_to_chainerx(self, backend_config):
        device = chainerx.get_device('native:0')
        arrays = self.get_dict_arrays_to_concat(backend_config)
        self.check_concat_dicts(
            arrays, device, backend.ChainerxDevice(device))


@testing.backend.inject_backend_tests(
    None,
    [
        # NumPy
        {},
        # CuPy
        {'use_cuda': True, 'cuda_device': 0},
        {'use_cuda': True, 'cuda_device': 1},
        # ChainerX
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
    ])
class TestConcatExamplesWithPadding(unittest.TestCase):

    def test_concat_arrays_padding(self, backend_config):
        arrays = backend_config.get_array(
            [numpy.random.rand(3, 4),
             numpy.random.rand(2, 5),
             numpy.random.rand(4, 3)])
        array = dataset.concat_examples(arrays, padding=0)
        self.assertEqual(array.shape, (3, 4, 5))
        self.assertEqual(type(array), type(arrays[0]))

        arrays = [backend.CpuDevice().send(a) for a in arrays]
        array = backend.CpuDevice().send(array)
        numpy.testing.assert_array_equal(array[0, :3, :4], arrays[0])
        numpy.testing.assert_array_equal(array[0, 3:, :], 0)
        numpy.testing.assert_array_equal(array[0, :, 4:], 0)
        numpy.testing.assert_array_equal(array[1, :2, :5], arrays[1])
        numpy.testing.assert_array_equal(array[1, 2:, :], 0)
        numpy.testing.assert_array_equal(array[2, :4, :3], arrays[2])
        numpy.testing.assert_array_equal(array[2, :, 3:], 0)

    def test_concat_tuples_padding(self, backend_config):
        tuples = [
            backend_config.get_array(
                (numpy.random.rand(3, 4), numpy.random.rand(2, 5))),
            backend_config.get_array(
                (numpy.random.rand(4, 4), numpy.random.rand(3, 4))),
            backend_config.get_array(
                (numpy.random.rand(2, 5), numpy.random.rand(2, 6))),
        ]
        arrays = dataset.concat_examples(tuples, padding=0)
        self.assertEqual(len(arrays), 2)
        self.assertEqual(arrays[0].shape, (3, 4, 5))
        self.assertEqual(arrays[1].shape, (3, 3, 6))
        self.assertEqual(type(arrays[0]), type(tuples[0][0]))
        self.assertEqual(type(arrays[1]), type(tuples[0][1]))

        for i in range(len(tuples)):
            tuples[i] = (
                backend.CpuDevice().send(tuples[i][0]),
                backend.CpuDevice().send(tuples[i][1]))
        arrays = tuple(backend.CpuDevice().send(array) for array in arrays)
        numpy.testing.assert_array_equal(arrays[0][0, :3, :4], tuples[0][0])
        numpy.testing.assert_array_equal(arrays[0][0, 3:, :], 0)
        numpy.testing.assert_array_equal(arrays[0][0, :, 4:], 0)
        numpy.testing.assert_array_equal(arrays[0][1, :4, :4], tuples[1][0])
        numpy.testing.assert_array_equal(arrays[0][1, :, 4:], 0)
        numpy.testing.assert_array_equal(arrays[0][2, :2, :5], tuples[2][0])
        numpy.testing.assert_array_equal(arrays[0][2, 2:, :], 0)
        numpy.testing.assert_array_equal(arrays[1][0, :2, :5], tuples[0][1])
        numpy.testing.assert_array_equal(arrays[1][0, 2:, :], 0)
        numpy.testing.assert_array_equal(arrays[1][0, :, 5:], 0)
        numpy.testing.assert_array_equal(arrays[1][1, :3, :4], tuples[1][1])
        numpy.testing.assert_array_equal(arrays[1][1, 3:, :], 0)
        numpy.testing.assert_array_equal(arrays[1][1, :, 4:], 0)
        numpy.testing.assert_array_equal(arrays[1][2, :2, :6], tuples[2][1])
        numpy.testing.assert_array_equal(arrays[1][2, 2:, :], 0)

    def test_concat_dicts_padding(self, backend_config):
        dicts = [
            {'x': numpy.random.rand(3, 4), 'y': numpy.random.rand(2, 5)},
            {'x': numpy.random.rand(4, 4), 'y': numpy.random.rand(3, 4)},
            {'x': numpy.random.rand(2, 5), 'y': numpy.random.rand(2, 6)},
        ]
        dicts = [
            {key: backend_config.get_array(arr) for key, arr in d.items()}
            for d in dicts]
        arrays = dataset.concat_examples(dicts, padding=0)
        self.assertIn('x', arrays)
        self.assertIn('y', arrays)
        self.assertEqual(arrays['x'].shape, (3, 4, 5))
        self.assertEqual(arrays['y'].shape, (3, 3, 6))
        self.assertEqual(type(arrays['x']), type(dicts[0]['x']))
        self.assertEqual(type(arrays['y']), type(dicts[0]['y']))

        for d in dicts:
            d['x'] = backend.CpuDevice().send(d['x'])
            d['y'] = backend.CpuDevice().send(d['y'])
        arrays = {
            'x': backend.CpuDevice().send(arrays['x']),
            'y': backend.CpuDevice().send(arrays['y'])}
        numpy.testing.assert_array_equal(arrays['x'][0, :3, :4], dicts[0]['x'])
        numpy.testing.assert_array_equal(arrays['x'][0, 3:, :], 0)
        numpy.testing.assert_array_equal(arrays['x'][0, :, 4:], 0)
        numpy.testing.assert_array_equal(arrays['x'][1, :4, :4], dicts[1]['x'])
        numpy.testing.assert_array_equal(arrays['x'][1, :, 4:], 0)
        numpy.testing.assert_array_equal(arrays['x'][2, :2, :5], dicts[2]['x'])
        numpy.testing.assert_array_equal(arrays['x'][2, 2:, :], 0)
        numpy.testing.assert_array_equal(arrays['y'][0, :2, :5], dicts[0]['y'])
        numpy.testing.assert_array_equal(arrays['y'][0, 2:, :], 0)
        numpy.testing.assert_array_equal(arrays['y'][0, :, 5:], 0)
        numpy.testing.assert_array_equal(arrays['y'][1, :3, :4], dicts[1]['y'])
        numpy.testing.assert_array_equal(arrays['y'][1, 3:, :], 0)
        numpy.testing.assert_array_equal(arrays['y'][1, :, 4:], 0)
        numpy.testing.assert_array_equal(arrays['y'][2, :2, :6], dicts[2]['y'])
        numpy.testing.assert_array_equal(arrays['y'][2, 2:, :], 0)


@testing.parameterize(
    {'padding': None},
    {'padding': 0},
)
class TestConcatExamplesWithBuiltInTypes(unittest.TestCase):

    int_arrays = [1, 2, 3]
    float_arrays = [1.0, 2.0, 3.0]

    def check_device(self, array, device, expected_device):
        self.assertIsInstance(array, expected_device.xp.ndarray)
        self.assertEqual(
            backend.get_device_from_array(array), expected_device)

    def check_concat_arrays(
            self, arrays, device, expected_device, expected_dtype):
        array = dataset.concat_examples(arrays, device, self.padding)
        self.assertEqual(array.shape, (len(arrays),))
        self.check_device(array, device, expected_device)

        np_array = backend.CpuDevice().send(array)
        for x, y in zip(np_array, arrays):
            assert x.dtype == expected_dtype
            numpy.testing.assert_array_equal(
                x, numpy.array(y, dtype=expected_dtype))

    def test_concat_arrays_to_cpu(self):
        if sys.platform == 'win32':
            expected_int_dtype = numpy.int32
        else:
            expected_int_dtype = numpy.int64
        for device in (-1, None):
            self.check_concat_arrays(
                self.int_arrays,
                device,
                backend.CpuDevice(),
                expected_int_dtype)
            self.check_concat_arrays(
                self.float_arrays,
                device,
                backend.CpuDevice(),
                numpy.float64)

    @attr.gpu
    def test_concat_arrays_to_gpu(self):
        device = 0
        if sys.platform == 'win32':
            expected_int_dtype = numpy.int32
        else:
            expected_int_dtype = numpy.int64
        self.check_concat_arrays(
            self.int_arrays,
            device,
            backend.GpuDevice.from_device_id(0),
            expected_int_dtype)
        self.check_concat_arrays(
            self.float_arrays,
            device,
            backend.GpuDevice.from_device_id(0),
            numpy.float64)

    @attr.chainerx
    def test_concat_arrays_to_chainerx(self):
        device = 'native:0'
        self.check_concat_arrays(
            self.int_arrays,
            device,
            backend.ChainerxDevice(chainerx.get_device(device)),
            numpy.int64)
        self.check_concat_arrays(
            self.float_arrays,
            device,
            backend.ChainerxDevice(chainerx.get_device(device)),
            numpy.float64)


def get_xp(gpu):
    if gpu:
        return cuda.cupy
    else:
        return numpy


@testing.parameterize(
    {'device': None, 'src_gpu': False, 'dst_gpu': False},
    {'device': -1, 'src_gpu': False, 'dst_gpu': False},
)
class TestToDeviceCPU(unittest.TestCase):

    def test_to_device(self):
        src_xp = get_xp(self.src_gpu)
        dst_xp = get_xp(self.dst_gpu)
        x = src_xp.array([1], 'i')
        y = dataset.to_device(self.device, x)
        self.assertIsInstance(y, dst_xp.ndarray)


@testing.parameterize(
    {'device': None, 'src_gpu': True, 'dst_gpu': True},

    {'device': -1, 'src_gpu': True, 'dst_gpu': False},

    {'device': 0, 'src_gpu': False, 'dst_gpu': True},
    {'device': 0, 'src_gpu': True, 'dst_gpu': True},
)
class TestToDeviceGPU(unittest.TestCase):

    @attr.gpu
    def test_to_device(self):
        src_xp = get_xp(self.src_gpu)
        dst_xp = get_xp(self.dst_gpu)
        x = src_xp.array([1], 'i')
        y = dataset.to_device(self.device, x)
        self.assertIsInstance(y, dst_xp.ndarray)

        if self.device is not None and self.device >= 0:
            self.assertEqual(int(y.device), self.device)

        if self.device is None and self.src_gpu:
            self.assertEqual(int(x.device), int(y.device))


@testing.parameterize(
    {'device': 1, 'src_gpu': False, 'dst_gpu': True},
    {'device': 1, 'src_gpu': True, 'dst_gpu': True},
)
class TestToDeviceMultiGPU(unittest.TestCase):

    @attr.multi_gpu(2)
    def test_to_device(self):
        src_xp = get_xp(self.src_gpu)
        dst_xp = get_xp(self.dst_gpu)
        x = src_xp.array([1], 'i')
        y = dataset.to_device(self.device, x)
        self.assertIsInstance(y, dst_xp.ndarray)

        self.assertEqual(int(y.device), self.device)


testing.run_module(__name__, __file__)
