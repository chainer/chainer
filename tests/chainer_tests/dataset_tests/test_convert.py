import unittest

import numpy

from chainer import backend
from chainer.backends import cuda
from chainer import dataset
from chainer import testing
from chainer.testing import attr
import chainerx


class TestConcatExamples(unittest.TestCase):

    def get_arrays_to_concat(self, conv):
        np_arrays = [numpy.random.rand(2, 3) for _ in range(5)]
        return [conv(a) for a in np_arrays]

    def check_device(self, array, expected_device):
        # TODO(niboshi): This is a workaround because get_device_from_array
        # returns chainerx.DeviceScope instead of chainerx.Device.
        if chainerx.is_available() and isinstance(array, chainerx.ndarray):
            assert array.device == expected_device
        else:
            assert backend.get_device_from_array(array) == expected_device

    def check_concat_arrays(self, arrays, device, expected_device):
        array = dataset.concat_examples(arrays, device)
        self.assertEqual(array.shape, (len(arrays),) + arrays[0].shape)

        self.check_device(array, expected_device)

        np_array = backend.to_numpy(array)
        for x, y in zip(np_array, arrays):
            numpy.testing.assert_array_equal(x, backend.to_numpy(y))

    def test_concat_arrays_cpu(self):
        arrays = self.get_arrays_to_concat(lambda a: a)
        self.check_concat_arrays(arrays, None, cuda.DummyDevice)

    @attr.gpu
    def test_concat_arrays_gpu(self):
        arrays = self.get_arrays_to_concat(cuda.to_gpu)
        self.check_concat_arrays(arrays, None, cuda.Device(0))

    @attr.gpu
    def test_concat_arrays_numpy_to_gpu(self):
        arrays = self.get_arrays_to_concat(lambda a: a)
        self.check_concat_arrays(arrays, 0, cuda.Device(0))

    @attr.chainerx
    def test_concat_arrays_chainerx_native(self):
        arrays = self.get_arrays_to_concat(backend.to_chainerx)
        self.check_concat_arrays(arrays, None, chainerx.get_device('native:0'))

    @attr.chainerx
    @attr.gpu
    def test_concat_arrays_chainerx_cuda(self):
        arrays = self.get_arrays_to_concat(
            lambda a: backend.to_chainerx(cuda.to_gpu(a)))
        self.check_concat_arrays(arrays, None, chainerx.get_device('cuda:0'))

    @attr.chainerx
    def test_concat_arrays_numpy_to_chainerx_device(self):
        device = chainerx.get_device('native:0')
        arrays = self.get_arrays_to_concat(lambda a: a)
        self.check_concat_arrays(arrays, device, device)

    def get_tuple_arrays_to_concat(self, conv):
        return [(conv(numpy.random.rand(2, 3)), conv(numpy.random.rand(3, 4)))
                for _ in range(5)]

    def check_concat_tuples(self, tuples, device, expected_device):
        arrays = dataset.concat_examples(tuples, device)
        self.assertEqual(len(arrays), len(tuples[0]))
        for i in range(len(arrays)):
            shape = (len(tuples),) + tuples[0][i].shape
            self.assertEqual(arrays[i].shape, shape)

            self.check_device(arrays[i], expected_device)

            arr = backend.to_numpy(arrays[i])
            for x, y in zip(arr, tuples):
                numpy.testing.assert_array_equal(x, backend.to_numpy(y[i]))

    def test_concat_tuples_cpu(self):
        tuples = self.get_tuple_arrays_to_concat(lambda a: a)
        self.check_concat_tuples(tuples, None, cuda.DummyDevice)

    @attr.gpu
    def test_concat_tuples_gpu(self):
        tuples = self.get_tuple_arrays_to_concat(cuda.to_gpu)
        self.check_concat_tuples(tuples, None, cuda.Device(0))

    @attr.gpu
    def test_concat_tuples_numpy_to_gpu(self):
        tuples = self.get_tuple_arrays_to_concat(lambda a: a)
        self.check_concat_tuples(tuples, 0, cuda.Device(0))

    @attr.chainerx
    def test_concat_tuples_chainerx_native(self):
        arrays = self.get_tuple_arrays_to_concat(backend.to_chainerx)
        self.check_concat_tuples(arrays, None, chainerx.get_device('native:0'))

    @attr.chainerx
    @attr.gpu
    def test_concat_tuples_chainerx_cuda(self):
        arrays = self.get_tuple_arrays_to_concat(
            lambda a: backend.to_chainerx(cuda.to_gpu(a)))
        self.check_concat_tuples(arrays, None, chainerx.get_device('cuda:0'))

    @attr.chainerx
    def test_concat_tuples_numpy_to_chainerx_device(self):
        device = chainerx.get_device('native:0')
        arrays = self.get_tuple_arrays_to_concat(lambda a: a)
        self.check_concat_tuples(arrays, device, device)

    def get_dict_arrays_to_concat(self, conv):
        return [
            {'x': conv(numpy.random.rand(2, 3)),
             'y': conv(numpy.random.rand(3, 4))}
            for _ in range(5)]

    def check_concat_dicts(self, dicts, device, expected_device):
        arrays = dataset.concat_examples(dicts, device)
        self.assertEqual(frozenset(arrays.keys()), frozenset(dicts[0].keys()))
        for key in arrays:
            shape = (len(dicts),) + dicts[0][key].shape
            self.assertEqual(arrays[key].shape, shape)
            self.check_device(arrays[key], expected_device)

            arr = backend.to_numpy(arrays[key])
            for x, y in zip(arr, dicts):
                numpy.testing.assert_array_equal(x, backend.to_numpy(y[key]))

    def test_concat_dicts_cpu(self):
        dicts = self.get_dict_arrays_to_concat(lambda a: a)
        self.check_concat_dicts(dicts, None, cuda.DummyDevice)

    @attr.gpu
    def test_concat_dicts_gpu(self):
        dicts = self.get_dict_arrays_to_concat(cuda.to_gpu)
        self.check_concat_dicts(dicts, None, cuda.Device(0))

    @attr.gpu
    def test_concat_dicts_numpy_to_gpu(self):
        dicts = self.get_dict_arrays_to_concat(lambda a: a)
        self.check_concat_dicts(dicts, 0, cuda.Device(0))

    @attr.chainerx
    def test_concat_dicts_chainerx_native(self):
        arrays = self.get_dict_arrays_to_concat(backend.to_chainerx)
        self.check_concat_dicts(arrays, None, chainerx.get_device('native:0'))

    @attr.chainerx
    @attr.gpu
    def test_concat_dicts_chainerx_cuda(self):
        arrays = self.get_dict_arrays_to_concat(
            lambda a: backend.to_chainerx(cuda.to_gpu(a)))
        self.check_concat_dicts(arrays, None, chainerx.get_device('cuda:0'))

    @attr.chainerx
    def test_concat_dicts_numpy_to_chainerx_device(self):
        device = chainerx.get_device('native:0')
        arrays = self.get_dict_arrays_to_concat(lambda a: a)
        self.check_concat_dicts(arrays, device, device)


# TODO(niboshi): Add ChainerX tests
class TestConcatExamplesWithPadding(unittest.TestCase):

    def check_concat_arrays_padding(self, xp):
        arrays = [xp.random.rand(3, 4),
                  xp.random.rand(2, 5),
                  xp.random.rand(4, 3)]
        array = dataset.concat_examples(arrays, padding=0)
        self.assertEqual(array.shape, (3, 4, 5))
        self.assertEqual(type(array), type(arrays[0]))

        arrays = [cuda.to_cpu(a) for a in arrays]
        array = cuda.to_cpu(array)
        numpy.testing.assert_array_equal(array[0, :3, :4], arrays[0])
        numpy.testing.assert_array_equal(array[0, 3:, :], 0)
        numpy.testing.assert_array_equal(array[0, :, 4:], 0)
        numpy.testing.assert_array_equal(array[1, :2, :5], arrays[1])
        numpy.testing.assert_array_equal(array[1, 2:, :], 0)
        numpy.testing.assert_array_equal(array[2, :4, :3], arrays[2])
        numpy.testing.assert_array_equal(array[2, :, 3:], 0)

    def test_concat_arrays_padding_cpu(self):
        self.check_concat_arrays_padding(numpy)

    @attr.gpu
    def test_concat_arrays_padding_gpu(self):
        self.check_concat_arrays_padding(cuda.cupy)

    def check_concat_tuples_padding(self, xp):
        tuples = [
            (xp.random.rand(3, 4), xp.random.rand(2, 5)),
            (xp.random.rand(4, 4), xp.random.rand(3, 4)),
            (xp.random.rand(2, 5), xp.random.rand(2, 6)),
        ]
        arrays = dataset.concat_examples(tuples, padding=0)
        self.assertEqual(len(arrays), 2)
        self.assertEqual(arrays[0].shape, (3, 4, 5))
        self.assertEqual(arrays[1].shape, (3, 3, 6))
        self.assertEqual(type(arrays[0]), type(tuples[0][0]))
        self.assertEqual(type(arrays[1]), type(tuples[0][1]))

        for i in range(len(tuples)):
            tuples[i] = cuda.to_cpu(tuples[i][0]), cuda.to_cpu(tuples[i][1])
        arrays = tuple(cuda.to_cpu(array) for array in arrays)
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

    def test_concat_tuples_padding_cpu(self):
        self.check_concat_tuples_padding(numpy)

    @attr.gpu
    def test_concat_tuples_padding_gpu(self):
        self.check_concat_tuples_padding(cuda.cupy)

    def check_concat_dicts_padding(self, xp):
        dicts = [
            {'x': xp.random.rand(3, 4), 'y': xp.random.rand(2, 5)},
            {'x': xp.random.rand(4, 4), 'y': xp.random.rand(3, 4)},
            {'x': xp.random.rand(2, 5), 'y': xp.random.rand(2, 6)},
        ]
        arrays = dataset.concat_examples(dicts, padding=0)
        self.assertIn('x', arrays)
        self.assertIn('y', arrays)
        self.assertEqual(arrays['x'].shape, (3, 4, 5))
        self.assertEqual(arrays['y'].shape, (3, 3, 6))
        self.assertEqual(type(arrays['x']), type(dicts[0]['x']))
        self.assertEqual(type(arrays['y']), type(dicts[0]['y']))

        for d in dicts:
            d['x'] = cuda.to_cpu(d['x'])
            d['y'] = cuda.to_cpu(d['y'])
        arrays = {'x': cuda.to_cpu(arrays['x']), 'y': cuda.to_cpu(arrays['y'])}
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

    def test_concat_dicts_padding_cpu(self):
        self.check_concat_dicts_padding(numpy)

    @attr.gpu
    def test_concat_dicts_padding_gpu(self):
        self.check_concat_dicts_padding(cuda.cupy)


# TODO(niboshi): Add ChainerX tests
@testing.parameterize(
    {'padding': None},
    {'padding': 0},
)
class TestConcatExamplesWithBuiltInTypes(unittest.TestCase):

    int_arrays = [1, 2, 3]
    float_arrays = [1.0, 2.0, 3.0]

    def check_device(self, array, device):
        if device is not None and device >= 0:
            self.assertIsInstance(array, cuda.ndarray)
            self.assertEqual(array.device.id, device)
        else:
            self.assertIsInstance(array, numpy.ndarray)

    def check_concat_arrays(self, arrays, device, expected_type):
        array = dataset.concat_examples(arrays, device, self.padding)
        self.assertEqual(array.shape, (len(arrays),))
        self.check_device(array, device)

        for x, y in zip(array, arrays):
            if backend.get_array_module(x) == numpy:
                numpy.testing.assert_array_equal(
                    numpy.array(x),
                    numpy.array(y, dtype=expected_type))
            else:
                numpy.testing.assert_array_equal(
                    cuda.to_cpu(x),
                    numpy.array(y, dtype=expected_type))

    def test_concat_arrays_cpu(self):
        for device in (-1, None):
            self.check_concat_arrays(self.int_arrays,
                                     device=device,
                                     expected_type=numpy.int64)
            self.check_concat_arrays(self.float_arrays,
                                     device=device,
                                     expected_type=numpy.float64)

    @attr.gpu
    def test_concat_arrays_gpu(self):
        self.check_concat_arrays(self.int_arrays,
                                 device=cuda.Device().id,
                                 expected_type=numpy.int64)
        self.check_concat_arrays(self.float_arrays,
                                 device=cuda.Device().id,
                                 expected_type=numpy.float64)


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
