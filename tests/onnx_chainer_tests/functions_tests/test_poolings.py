import chainer
import chainer.functions as F
from chainer import testing
import numpy as np

from onnx_chainer.testing import input_generator
from onnx_chainer_tests.helper import ONNXModelTest


@testing.parameterize(
    {'op_name': 'average_pooling_2d',
     'in_shape': (1, 3, 6, 6), 'args': [2, 1, 0], 'cover_all': None},
    {'op_name': 'average_pooling_2d', 'condition': 'pad1',
     'in_shape': (1, 3, 6, 6), 'args': [3, 2, 1], 'cover_all': None},
    {'op_name': 'average_pooling_nd',
     'in_shape': (1, 3, 6, 6, 6), 'args': [2, 1, 1], 'cover_all': None},
    {'op_name': 'max_pooling_2d',
     'in_shape': (1, 3, 6, 6), 'args': [2, 1, 1], 'cover_all': False},
    {'op_name': 'max_pooling_2d', 'condition': 'coverall',
     'in_shape': (1, 3, 6, 5), 'args': [3, (2, 1), 1], 'cover_all': True},
    {'op_name': 'max_pooling_nd',
     'in_shape': (1, 3, 6, 6, 6), 'args': [2, 1, 1], 'cover_all': False},
    {'op_name': 'max_pooling_nd', 'condition': 'coverall',
     'in_shape': (1, 3, 6, 5, 4), 'args': [3, 2, 1], 'cover_all': True},
    {'op_name': 'unpooling_2d',
     'in_shape': (1, 3, 6, 6), 'args': [3, None, 0], 'cover_all': False},
    # TODO(disktnk): when cover_all=True, interpolation between Chainer and
    # ONNXRuntime does not match, so skip output value check.
    {'op_name': 'unpooling_2d', 'condition': 'coverall',
     'in_shape': (1, 3, 6, 6), 'args': [3, None, 0], 'cover_all': True,
     'skip_check_ver': True},
)
class TestPoolings(ONNXModelTest):

    def setUp(self):
        ops = getattr(F, self.op_name)
        self.model = Model(ops, self.args, self.cover_all)
        self.x = input_generator.increasing(*self.in_shape)

    def test_output(self):
        name = self.op_name
        if hasattr(self, 'condition'):
            name += '_' + self.condition
        skip_out_check = getattr(self, 'skip_check_ver', None)
        if skip_out_check is not None:
            skip_out_check = self.target_opsets
        self.expect(
            self.model, self.x, name=name,
            skip_outvalue_version=skip_out_check, expected_num_initializers=0)


class Model(chainer.Chain):

    def __init__(self, ops, args, cover_all):
        super(Model, self).__init__()
        self.ops = ops
        self.args = args
        self.cover_all = cover_all

    def __call__(self, x):
        if self.cover_all is not None:
            return self.ops(*([x] + self.args), cover_all=self.cover_all)
        else:
            return self.ops(*([x] + self.args))


class TestROIPooling2D(ONNXModelTest):

    def setUp(self):
        # these parameters are referenced from chainer test
        in_shape = (3, 3, 12, 8)
        self.x = input_generator.positive_increasing(*in_shape)
        # In chainer test, x is shuffled and normalize-like conversion,
        # In this test, those operations are skipped.
        # If x includes negative value, not match with onnxruntime output.
        # You can reproduce this issue by changing `positive_increasing` to
        # `increase`
        self.rois = np.array([
            [0, 1, 1, 6, 6],
            [2, 6, 2, 7, 11],
            [1, 3, 1, 5, 10],
            [0, 3, 3, 3, 3]], dtype=np.float32)
        kwargs = {
            'outh': 3,
            'outw': 7,
            'spatial_scale': 0.6
        }

        class Model(chainer.Chain):
            def __init__(self, kwargs):
                super(Model, self).__init__()
                self.kwargs = kwargs

            def __call__(self, x, rois):
                return F.roi_pooling_2d(x, rois, **self.kwargs)

        self.model = Model(kwargs)

    def test_output(self):
        with testing.assert_warns(UserWarning):
            self.expect(self.model, [self.x, self.rois])
