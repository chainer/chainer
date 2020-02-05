import chainer
import chainer.functions as F
from chainer import testing
import numpy as np
import onnx
import pytest

from onnx_chainer import export
from onnx_chainer.testing import input_generator
from onnx_chainer_tests.helper import ONNXModelChecker
from onnx_chainer_tests.helper import ONNXModelTest


@testing.parameterize(
    # cast
    # {'ops': 'cast', 'input_shape': (1, 5),
    #  'input_argname': 'x',
    #  'args': {'typ': np.float16}},
    {'ops': 'cast', 'input_shape': (1, 5),
     'input_argname': 'x',
     'args': {'typ': np.float64}},

    # depth2space
    {'ops': 'depth2space', 'input_shape': (1, 12, 6, 6),
     'input_argname': 'X',
     'args': {'r': 2}},

    # pad
    {'ops': 'pad', 'input_shape': (1, 2, 3, 4),
     'input_argname': 'x',
     'args': {'pad_width': ((0, 0), (0, 0), (2, 2), (2, 2)),
              'mode': 'constant'},
     'name': 'pad_constant'},
    {'ops': 'pad', 'input_shape': (1, 2, 3, 4),
     'input_argname': 'x',
     'args': {'pad_width': ((0, 0), (0, 0), (2, 2), (2, 2)),
              'mode': 'reflect'},
     'name': 'pad_reflect'},
    {'ops': 'pad', 'input_shape': (1, 2, 3, 4),
     'input_argname': 'x',
     'args': {'pad_width': ((0, 0), (0, 0), (2, 2), (2, 2)),
              'mode': 'edge'},
     'name': 'pad_edge'},
    {'ops': 'pad', 'input_shape': (1, 2, 3, 4),
     'input_argname': 'x',
     'args': {'pad_width': ((1, 3), (2, 0), (7, 1), (4, 4)),
              'mode': 'constant'},
     'name': 'pad_imbalance_pad_width'},
    {'ops': 'pad', 'input_shape': (1, 2, 3, 4),
     'input_argname': 'x',
     'args': {'pad_width': ((0, 0), (0, 0), (2, 2), (2, 2)),
              'mode': 'constant',
              'constant_values': -1},
     'name': 'pad_with_constant_values'},
    {'ops': 'pad', 'input_shape': (1, 2, 3, 4),
     'input_argname': 'x',
     'args': {'pad_width': 2,
              'mode': 'constant'},
     'name': 'pad_scalar_pad_width'},

    # reshape
    {'ops': 'reshape', 'input_shape': (1, 6),
     'input_argname': 'x',
     'args': {'shape': (1, 2, 1, 3)}},

    # space2depth
    {'ops': 'space2depth', 'input_shape': (1, 12, 6, 6),
     'input_argname': 'X',
     'args': {'r': 2}},

    # split_axis
    {'ops': 'split_axis', 'input_shape': (1, 6),
     'input_argname': 'x',
     'args': {'indices_or_sections': 2,
              'axis': 1, 'force_tuple': True},
     'name': 'split_axis_force_tuple_true'},
    {'ops': 'split_axis', 'input_shape': (1, 6),
     'input_argname': 'x',
     'args': {'indices_or_sections': 2,
              'axis': 1, 'force_tuple': False},
     'name': 'split_axis_force_tuple_false'},
    {'ops': 'split_axis', 'input_shape': (1, 6),
     'input_argname': 'x',
     'args': {'indices_or_sections': [1, 2], 'axis': 1},
     'name': 'split_axis_list'},

    # squeeze
    {'ops': 'squeeze', 'input_shape': (1, 3, 1, 2),
     'input_argname': 'x',
     'args': {'axis': None},
     'name': 'squeeze_axis_none'},
    {'ops': 'squeeze', 'input_shape': (1, 3, 1, 2, 1),
     'input_argname': 'x',
     'args': {'axis': (2, 4)}},

    # swapaxes
    {'ops': 'swapaxes', 'input_shape': (2, 3, 4, 5),
     'input_argname': 'x',
     'args': {'axis1': 1, 'axis2': 2}},
    {'ops': 'swapaxes', 'input_shape': (2, 3, 4, 5),
     'input_argname': 'x',
     'args': {'axis1': -3, 'axis2': -1}},

    # tile
    {'ops': 'tile', 'input_shape': (1, 5),
     'input_argname': 'x',
     'args': {'reps': (1, 2)}},

    # transpose
    {'ops': 'transpose', 'input_shape': (1, 5),
     'input_argname': 'x',
     'args': {'axes': None}},

    # copy
    {'ops': 'copy', 'input_shape': (1, 5),
     'input_argname': 'x',
     'args': {'dst': -1}},

    # get_item
    {'ops': 'get_item', 'input_shape': (2, 2, 3),
     'input_argname': 'x',
     'args': {'slices': slice(0, 2)},
     'name': 'get_item_0to2'},
    {'ops': 'get_item', 'input_shape': (2, 2, 3),
     'input_argname': 'x',
     'args': {'slices': (slice(1))},
     'name': 'get_item_to1'},
    {'ops': 'get_item', 'input_shape': (2, 2, 3),
     'input_argname': 'x',
     'args': {'slices': (slice(1, None))},
     'name': 'get_item_1tonone'},
    {'ops': 'get_item', 'input_shape': (2, 2, 3),
     'input_argname': 'x',
     'args': {'slices': 0},
     'name': 'get_item_0'},
    {'ops': 'get_item', 'input_shape': (2, 2, 3),
     'input_argname': 'x',
     'args': {'slices': -1},
     'name': 'get_item_minus_1'},
    {'ops': 'get_item', 'input_shape': (2, 2, 3),
     'input_argname': 'x',
     'args': {'slices': np.array(0)},
     'name': 'get_item_npscalar0'},
    {'ops': 'get_item', 'input_shape': (2, 2, 3),
     'input_argname': 'x',
     'args': {'slices': (None, slice(0, 2))},
     'name': 'get_item_none_0to2'},
    {'ops': 'get_item', 'input_shape': (2, 2, 3),
     'input_argname': 'x',
     'args': {'slices': (Ellipsis, slice(0, 2))},
     'name': 'get_item_ellipsis_0to2'},
    # get_item, combine newaxis, slice, single index, ellipsis
    {'ops': 'get_item', 'input_shape': (2, 2, 3, 3, 3, 4),
     'input_argname': 'x',
     'args': {'slices': (0, None, Ellipsis, 0, None, slice(0, 2), None, 0)},
     'name': 'get_item_complicated'},
    {'ops': 'get_item', 'input_shape': (2, 2, 3),
     'input_argname': 'x',
     'args': {'slices': (slice(None), slice(0, 1), slice(None, 2))},
     'name': 'get_item_start_from_none'},

    # expand_dims
    {'ops': 'expand_dims', 'input_shape': (3,),
     'input_argname': 'x', 'args': {'axis': 0},
     'name': 'expand_dims_0'},
    {'ops': 'expand_dims', 'input_shape': (3,),
     'input_argname': 'x', 'args': {'axis': 1},
     'name': 'expand_dims_1'},
    {'ops': 'expand_dims', 'input_shape': (3,),
     'input_argname': 'x', 'args': {'axis': -2},
     'name': 'expand_dims_minus2'},

    # repeat
    {'ops': 'repeat', 'input_shape': (3,),
     'input_argname': 'x', 'args': {'repeats': 2},
     'name': 'repeat_ndim1'},
    {'ops': 'repeat', 'input_shape': (2, 3),
     'input_argname': 'x', 'args': {'repeats': 2, 'axis': 1},
     'name': 'repeat_with_axis'},
    {'ops': 'repeat', 'input_shape': (2, 3),
     'input_argname': 'x', 'args': {'repeats': 2},
     'name': 'repeat_default_axis'},

    # separate
    {'ops': 'separate', 'input_shape': (2, 3),
     'input_argname': 'x', 'args': {}, 'name': 'separate_axis0'},
    {'ops': 'separate', 'input_shape': (2, 3),
     'input_argname': 'x', 'args': {'axis': 1}, 'name': 'separate_axis1'},
    {'ops': 'separate', 'input_shape': (1, 2, 3),
     'input_argname': 'x', 'args': {}, 'name': 'separate_single_output'},

    # moveaxis
    {'ops': 'moveaxis', 'input_shape': (2, 3, 4, 5),
     'input_argname': 'x', 'args': {'source': 0, 'destination': -1}},
    {'ops': 'moveaxis', 'input_shape': (2, 3, 4, 5),
     'input_argname': 'x', 'args': {'source': (0, 3), 'destination': (2, 0)}},

    # rollaxis
    {'ops': 'rollaxis', 'input_shape': (2, 3, 4, 5),
     'input_argname': 'x', 'args': {'axis': 2, 'start': 0}},
)
class TestArrayOperators(ONNXModelTest):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self, ops, args, input_argname):
                super(Model, self).__init__()
                self.ops = getattr(F, ops)
                self.args = args
                self.input_argname = input_argname

            def __call__(self, x):
                self.args[self.input_argname] = x
                return self.ops(**self.args)

        self.model = Model(self.ops, self.args, self.input_argname)
        self.x = input_generator.increasing(*self.input_shape)

    def test_output(self):
        name = self.ops
        if hasattr(self, 'name'):
            name = self.name
        self.expect(
            self.model, self.x, name=name, expected_num_initializers=0)


class TestGetItem(ONNXModelChecker):
    # When chainer.testing.parameterize is used with list or ndarray parameter,
    # it causes regex warning. To resolve, use pytest's parameterize.

    @pytest.mark.parametrize(
        'name,slices', [
            ('gather_axis0', ([[0, 1], [0, 1]],)),
            ('gather_axis1', (slice(None), [[0, 1], [1, 2]], slice(None))),
            ('gather_axis2', (slice(None), slice(None), [[0, 1], [1, 2]])),
            ('gather_ndarray', (
                Ellipsis, np.array([[0, 1], [1, 2]], dtype=np.int64))),
            ('gather_before_squeezed', (slice(None), 0, [[0, 1], [2, 3]])),
            ('gather_after_squeezed', (slice(None), [[0, 1], [1, 2]], 0)),
            ('gather_unsqueezed', (
                slice(None), None, [[0, 1], [1, 2]], slice(None))),
            ('gathernd', [[0, 1], [1, 2]]),
            ('gathernd_slice_none', [[0, 1], [0, 1], slice(None)]),
            ('gathernd_full_idx', [[0, 1], [0, 1], [2, 3]]),
            ('gathernd_before_slice', [0, [0, 1], [2, 3]]),
            ('gathernd_after_slice', [[0, 1], [0, 2], 0]),
            ('gathernd_unsqueezed', [[0, 1], [0, 2], None])
        ])
    def test_get_item_gather(self, name, slices):
        skip_opsets = None
        if name.startswith('gathernd'):
            skip_opsets = tuple(range(7, 11))
        name = 'get_item_' + name

        model = chainer.Sequential(
            lambda x: F.get_item(x, slices=slices))
        x = input_generator.increasing(2, 3, 4)

        self.expect(
            model, x, name=name, expected_num_initializers=0,
            skip_opset_version=skip_opsets)

    @pytest.mark.parametrize(
        'name,slices', [
            ('step1', [slice(1, None, 1)]),
            ('step2', [slice(None, None, None), slice(None, 4, 2)]),
            ('step_neg1', [slice(None, None, -1)]),
            ('step_neg2', [slice(None, None, None), slice(4, None, -2)]),
        ])
    def test_get_item_slice_step(self, name, slices):
        skip_opsets = tuple(range(7, 11))
        name = 'get_item_' + name

        model = chainer.Sequential(
            lambda x: F.get_item(x, slices=slices))
        x = input_generator.increasing(2, 3, 4)

        self.expect(
            model, x, name=name, expected_num_initializers=0,
            skip_opset_version=skip_opsets)


class TestGetItemError(object):

    @pytest.mark.parametrize('slices', [
        [[0, 1], [1, 2]], [slice(None, None, 2)]
    ])
    def test_get_item_unsupported(self, slices):
        model = chainer.Sequential(
            lambda x: F.get_item(x, slices=slices))
        x = input_generator.increasing(2, 3, 4)

        with pytest.raises(ValueError):
            export(model, x, opset_version=7)

    @pytest.mark.skipif(
        onnx.defs.onnx_opset_version() < 11, reason='not support GatherND')
    @pytest.mark.parametrize(
        'slices', [
            [[0, 1], 0, [0, 1]],
            [slice(None), [0, 1], [0, 1]],
            [None, [0, 1], [0, 1]]
        ]
    )
    def test_get_item_unsupported_advanced_index(self, slices):
        model = chainer.Sequential(
            lambda x: F.get_item(x, slices=slices))
        x = input_generator.increasing(2, 3, 4)

        with pytest.raises(ValueError):
            export(model, x)


class TestConcat(ONNXModelTest):

    def setUp(self):
        class Model(chainer.Chain):

            def __init__(self):
                super(Model, self).__init__()

            def __call__(self, x1, x2):
                return F.concat((x1, x2))

        self.model = Model()
        self.x1 = input_generator.increasing(2, 5)
        self.x2 = input_generator.increasing(2, 4)

    def test_output(self):
        self.expect(self.model, (self.x1, self.x2))


class TestWhere(ONNXModelTest):

    def test_output(self):
        model = chainer.Sequential(
            F.where
        )
        cond = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.bool)
        x = input_generator.increasing(2, 3)
        y = np.zeros((2, 3), np.float32)
        self.expect(model, (cond, x, y), skip_opset_version=[7, 8])


class TestResizeImages(ONNXModelTest):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self, ops, args, input_argname):
                super(Model, self).__init__()
                self.ops = ops
                self.args = args
                self.input_argname = input_argname

            def __call__(self, x):
                self.args[self.input_argname] = x
                return self.ops(**self.args)

        # (batch, channel, height, width) = (1, 1, 2, 2)
        self.x = np.array([[[[64, 32], [64, 32]]]], np.float32)

        # 2x upsampling
        args = {'output_shape': (4, 4)}
        self.model = Model(F.resize_images, args, 'x')

    def test_output(self):

        # FIXME(syoyo): Currently the test will fail due to the different
        # behavior of bilinear interpolation between Chainer and onnxruntime.
        # So disable output value check for a while.
        #
        # Currently Chainer will give [64, 53.333336, 42.666668, 32]
        # (same result with tensorflow r1.13.1 with `align_corners=True`),
        # while onnxruntime gives [64, 48, 32, 32]
        # (same result with tensorflow r1.13.1 with `align_corners=False`)
        #
        # However, the correct behavior will be [64, 54, 40, 32].
        # (cv2.resize and tensorflow master(r1.14 or r2.0) after this fix:
        #  https://github.com/tensorflow/tensorflow/issues/6720)

        self.check_out_values = None  # Skip output value check

        with testing.assert_warns(UserWarning):
            self.expect(self.model, self.x, expected_num_initializers=0)


@testing.parameterize(
    {'ops': 'stack', 'in_shapes': [(3, 4), (3, 4)], 'kwargs': {},
     'name': 'stack_default'},
    {'ops': 'stack', 'in_shapes': [(3, 4), (3, 4)], 'kwargs': {'axis': 1},
     'name': 'stack_axis1'},
    {'ops': 'stack', 'in_shapes': [(3, 4), (3, 4)], 'kwargs': {'axis': 2},
     'name': 'stack_axis2'},
    {'ops': 'stack', 'in_shapes': [(3, 4), (3, 4)], 'kwargs': {'axis': -1},
     'name': 'stack_axis_neg'},

    {'ops': 'vstack', 'inputs': [2, 3], 'kwargs': {},
     'name': 'vstack_ndim0'},
    {'ops': 'vstack', 'in_shapes': [(3,), (3,)], 'kwargs': {},
     'name': 'vstack_ndim1'},
    {'ops': 'vstack', 'in_shapes': [(3, 4), (2, 4)], 'kwargs': {},
     'name': 'vstack_ndim2'},

    {'ops': 'hstack', 'inputs': [2, 3], 'kwargs': {},
     'name': 'hstack_ndim0'},
    {'ops': 'hstack', 'in_shapes': [(3,), (3,)], 'kwargs': {},
     'name': 'hstack_ndim1'},
    {'ops': 'hstack', 'in_shapes': [(3, 4), (3, 2)], 'kwargs': {},
     'name': 'hstack_ndim2'},

    {'ops': 'dstack', 'inputs': [2, 3], 'kwargs': {},
     'name': 'dstack_ndim0'},
    {'ops': 'dstack', 'in_shapes': [(3,), (3,)], 'kwargs': {},
     'name': 'dstack_ndim1'},
    {'ops': 'dstack', 'in_shapes': [(3, 2), (3, 2)], 'kwargs': {},
     'name': 'dstack_ndim2'},
    {'ops': 'dstack', 'in_shapes': [(3, 2, 2), (3, 2, 1)], 'kwargs': {},
     'name': 'dstack_ndim3'},
)
class TestStack(ONNXModelTest):

    def test_output(self):

        class Model(chainer.Chain):
            def __init__(self, ops, kwargs):
                super(Model, self).__init__()
                self.ops = getattr(F, ops)
                self.kwargs = kwargs

            def __call__(self, *xs):
                return self.ops(xs, **self.kwargs)

        model = Model(ops=self.ops, kwargs=self.kwargs)
        if hasattr(self, 'inputs'):
            xs = [np.array(value, dtype=np.float32) for value in self.inputs]
        else:
            xs = [input_generator.increasing(*shape) for
                  shape in self.in_shapes]

        self.expect(model, xs, name=self.name)


class TestShape(ONNXModelTest):

    def test_output(self):
        from onnx_chainer.replace_func import as_funcnode

        class Model(chainer.Chain):
            def __init__(self):
                super().__init__()

            @as_funcnode('Shape')
            def shape(self, x):
                # ONNX Shape operator constrains to return int64 type
                return np.array(x.shape, dtype=np.int64)

            def forward(self, x):
                # use shape method instead of x.shape to connect graph.
                return self.shape(x)

        model = Model()
        x = input_generator.increasing(3, 4, 5)

        self.expect(model, (x,))


class TestDynamicReshape(ONNXModelTest):

    def test_output(self):
        from onnx_chainer.replace_func import as_funcnode

        class Model(chainer.Chain):
            def __init__(self):
                super().__init__()

            @as_funcnode('Reshape')
            def dynamic_reshape(self, x, shape):
                # shape is expected as variable type
                return F.reshape(x, tuple(shape.array))

            def forward(self, x, shape):
                return self.dynamic_reshape(x, shape)

        model = Model()
        x = input_generator.increasing(3, 4, 5)
        shape = np.array([12, 5], dtype=np.int64)

        def check_no_param(onnx_model, path):
            assert not any(['param' in v.name for v in onnx_model.graph.input])

        self.expect(model, (x, shape), custom_model_test_func=check_no_param)


@testing.parameterize(
    {'kwargs': {}, 'name': 'permutate'},
    {'kwargs': {'inv': True}, 'name': 'permutate_inv'},
    {'kwargs': {'axis': 1}, 'name': 'permutate_axis1'},
    {'kwargs': {'axis': 1, 'inv': True}, 'name': 'permutate_axis1_inv'},
)
class TestPermutate(ONNXModelTest):

    def test_output(self):

        class Model(chainer.Chain):
            def __init__(self, kwargs):
                super(Model, self).__init__()
                self.kwargs = kwargs

            def forward(self, x, indices):
                return F.permutate(x, indices, **self.kwargs)

        model = Model(kwargs=self.kwargs)

        x = np.arange(6).reshape((3, 2)).astype(np.float32)
        if self.kwargs.get('axis') == 1:
            indices = np.array([1, 0], np.int32)
        else:
            indices = np.array([2, 0, 1], np.int32)
        self.expect(model, (x, indices), name=self.name,
                    skip_opset_version=[7, 8])


@testing.parameterize(
    {'in_shapes': [(3, 4)], 'name': 'transpose_sequence_single_input'},
    {'in_shapes': [(1, 3), (1, 3)],
     'name': 'transpose_sequence_single_output'},
    {'in_shapes': [(2, 3), (2, 3), (2, 3), (2, 3)],
     'name': 'transpose_sequence_same_shape'},
)
class TestTransposeSequence(ONNXModelTest):

    def test_output(self):

        class Model(chainer.Chain):
            def __init__(self):
                super(Model, self).__init__()

            def __call__(self, *xs):
                return F.transpose_sequence(xs)

        model = Model()
        xs = [input_generator.increasing(*shape) for
              shape in self.in_shapes]

        self.expect(model, xs, name=self.name)


class TestSelectItem(ONNXModelTest):

    def test_output(self):

        class Model(chainer.Chain):
            def forward(self, x, t):
                return F.select_item(x, t)

        model = Model()
        x = input_generator.increasing(3, 5)
        t = np.array([4, 1, 0], dtype=np.int32)

        self.expect(
            model, (x, t), expected_num_initializers=0,
            skip_opset_version=list(range(1, 9)))
