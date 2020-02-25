import unittest

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import testing
import numpy as np
import pytest

from onnx_chainer import export
from onnx_chainer.export import RetainInputHook
from onnx_chainer.testing import input_generator
from onnx_chainer_tests.helper import ONNXModelTest


@testing.parameterize(
    {'condition': 'tuple'},
    {'condition': 'tuple_with_name', 'input_names': ['x', 'y', 'z']},
    {'condition': 'list', 'in_type': 'list'},
    {'condition': 'list_with_names', 'in_type': 'list',
     'input_names': ['x', 'y', 'z']},
    {'condition': 'var', 'in_type': 'variable'},
    {'condition': 'var_with_names', 'in_type': 'variable',
     'input_names': ['x', 'y', 'z']},
    {'condition': 'varlist', 'in_type': 'variable_list'},
    {'condition': 'varlist_with_names', 'in_type': 'variable_list',
     'input_names': ['x', 'y', 'z']},
    {'condition': 'dict', 'in_type': 'dict'},
    {'condition': 'dict_with_names', 'in_type': 'dict',
     'input_names': {'x': 'in_x', 'y': 'in_y', 'z': 'in_z'}},
    {'condition': 'dict_with_name_list', 'in_type': 'dict',
     'input_names': ['x', 'y', 'z']},
    {'condition': 'vardict', 'in_type': 'variable_dict'},
    {'condition': 'vardict_with_names', 'in_type': 'variable_dict',
     'input_names': {'x': 'in_x', 'y': 'in_y', 'z': 'in_z'}},
)
class TestMultipleInputs(ONNXModelTest):

    def get_model(self):

        class Model(chainer.Chain):

            def __init__(self):
                super(Model, self).__init__()
                with self.init_scope():
                    self.prelu = L.PReLU()

            def __call__(self, x, y, z):
                return F.relu(x) + self.prelu(y) * z

        return Model()

    def get_x(self, in_type=None):
        base_x = (input_generator.increasing(1, 5),
                  input_generator.increasing(1, 5)*1.1,
                  input_generator.increasing(1, 5)*1.2)
        names = ['x', 'y', 'z']
        if in_type is None:
            return base_x
        elif in_type == 'list':
            return list(base_x)
        elif in_type == 'variable':
            return tuple(chainer.Variable(v) for v in base_x)
        elif in_type == 'variable_list':
            return [chainer.Variable(v) for v in base_x]
        elif in_type == 'dict':
            return {names[i]: v for i, v in enumerate(base_x)}
        elif in_type == 'variable_dict':
            return {names[i]: chainer.Variable(v)
                    for i, v in enumerate(base_x)}

    def test_multiple_inputs(self):
        model = self.get_model()
        x = self.get_x(getattr(self, 'in_type', None))
        name = 'multipleinputs_' + self.condition
        input_names = getattr(self, 'input_names', None)
        self.expect(model, x, name=name, input_names=input_names)


class TestImplicitInput(ONNXModelTest):

    def test_implicit_param(self):
        class Model(chainer.Chain):

            def __init__(self):
                super(Model, self).__init__()
                self.frac = chainer.Parameter(np.array(2, dtype=np.float32))

            def forward(self, x):
                return x / self.frac

        x = chainer.Variable(np.array(1, dtype=np.float32))
        self.expect(Model(), x, name='implicit_param')

    def test_implicit_param_ndarray(self):
        class Model(chainer.Chain):

            def __init__(self):
                super(Model, self).__init__()
                self.frac = np.array(2, dtype=np.float32)

            def forward(self, x):
                return x / self.frac

        x = chainer.Variable(np.array(1, dtype=np.float32))
        self.expect(Model(), x, name='implicit_param_ndarray')

    def test_implicit_temporary_input(self):
        class Model(chainer.Chain):

            def forward(self, x):
                return x + chainer.Variable(np.array(3, dtype=np.float32))

        x = np.array(5, dtype=np.float32)
        self.expect(Model(), x, name='implicit_temp_input')

    def test_implicit_temporary_input_ndarray(self):
        class Model(chainer.Chain):

            def forward(self, x):
                return x + np.array(3, dtype=np.float32)

        x = np.array(5, dtype=np.float32)
        self.expect(Model(), x, name='implicit_temp_input_ndarray')


class TestRetainInputHook(object):

    def get_x(self, test_type):
        if test_type == 'list':
            return [
                chainer.Variable(np.array(3, dtype=np.float32)),
                chainer.Variable(np.array(5, dtype=np.float32))]
        elif test_type == 'dict':
            return {'x': chainer.Variable(np.array(3, dtype=np.float32))}
        elif test_type == 'array':
            return np.array(3, dtype=np.float32)
        else:
            assert test_type == 'variable'
            return chainer.Variable(np.array(3, dtype=np.float32))

    @pytest.mark.parametrize(
        'test_type', ['variable', 'list', 'dict', 'array'])
    def test_hook_for_funcnode(self, test_type):
        class Model(chainer.Chain):

            def forward(self, x):
                if test_type in ['variable', 'array']:
                    x = [chainer.as_variable(x)]
                elif test_type == 'dict':
                    x = list(x.values())
                x.append(chainer.Variable(np.array(7, np.float32)))
                return F.stack(x)

        model = Model()
        x = self.get_x(test_type)
        with RetainInputHook() as h:
            model(x)
        expected_count = 1
        if test_type == 'array':
            # input is ndarray and not checked in forward_preprocess
            expected_count += 1
        assert len(h.retain_inputs) == expected_count

    @pytest.mark.parametrize('test_type', ['array'])
    def test_hook_for_childlink(self, test_type):
        # TODO(disktnk): test_type='variable' is failed
        class ChildModel(chainer.Chain):

            def forward(self, x, h):
                if test_type in ['variable', 'array']:
                    h = [chainer.as_variable(h)]
                elif test_type == 'dict':
                    h = list(h.values())
                h.append(x)
                return F.stack(h)

        class ParentModel(chainer.Chain):

            def __init__(self, get_x):
                super().__init__()
                self.get_x = get_x
                with self.init_scope():
                    self.m = ChildModel()

            def forward(self, x):
                h = self.get_x(test_type)
                return self.m(x, h)

        model = ParentModel(self.get_x)
        x = self.get_x('variable')
        with RetainInputHook() as h:
            model(x)
        assert len(h.retain_inputs) == 1


@testing.parameterize(
    {'use_bn': True, 'out_type': 'dict', 'condition': 'bn_out_dict'},
    {'use_bn': False, 'out_type': 'dict', 'condition': 'out_dict'},
    {'use_bn': True, 'out_type': 'dict', 'condition': 'bn_out_dict_with_name',
     'output_names': {'tanh': 'out_tanh', 'sigmoid': 'out_sigmoid'}},
    {'use_bn': True, 'out_type': 'dict',
     'condition': 'bn_out_dict_with_name_list',
     'output_names': ('out_tanh', 'out_sigmoid')},
    {'use_bn': True, 'out_type': 'tuple', 'condition': 'bn_out_tuple'},
    {'use_bn': True, 'out_type': 'tuple',
     'condition': 'bn_out_tuple_with_name',
     'output_names': ['out_tanh', 'out_sigmoid']},
    {'use_bn': True, 'out_type': 'list', 'condition': 'bn_out_list'},
    {'use_bn': True, 'out_type': 'list', 'condition': 'bn_out_list_with_name',
     'output_names': ['out_tanh', 'out_sigmoid']},
)
class TestMultipleOutput(ONNXModelTest):

    def get_model(self, use_bn=False, out_type=None):
        class Model(chainer.Chain):

            def __init__(self, use_bn=False, out_type=None):
                super(Model, self).__init__()

                self._use_bn = use_bn
                self._out_type = out_type
                with self.init_scope():
                    self.conv = L.Convolution2D(None, 32, ksize=3, stride=1)
                    if self._use_bn:
                        self.bn = L.BatchNormalization(32)

            def __call__(self, x):
                h = self.conv(x)
                if self._use_bn:
                    h = self.bn(h)
                o1 = F.tanh(h)
                o2 = F.sigmoid(h)
                if self._out_type == 'dict':
                    return {
                        'tanh': o1,
                        'sigmoid': o2
                    }
                elif self._out_type == 'tuple':
                    return o1, o2
                elif self._out_type == 'list':
                    return [o1, o2]

        return Model(use_bn=use_bn, out_type=out_type)

    def test_multiple_outputs(self):
        model = self.get_model(use_bn=self.use_bn, out_type=self.out_type)
        x = np.zeros((1, 3, 32, 32), dtype=np.float32)
        name = 'multipleoutput_' + self.condition
        output_names = getattr(self, 'output_names', None)
        self.expect(model, x, name=name, output_names=output_names)


class TestIntermediateOutput(ONNXModelTest):

    def get_model(self):
        class Model(chainer.Chain):

            def __init__(self):
                super().__init__()
                with self.init_scope():
                    self.l1 = L.Linear(4)
                    self.l2 = L.Linear(5, initial_bias=0.1)

            def __call__(self, x):
                y = self.l1(x)
                z = self.l2(y)
                return y, z
        return Model()

    def test_outputs(self):
        model = self.get_model()
        x = np.ones((1, 3), dtype=np.float32)
        self.expect(model, x, output_names=['y', 'z'])


@testing.parameterize(
    {'out_kind': 'var'},
    {'out_kind': 'array'},
    {'out_kind': 'array_in_tuple'},
    {'out_kind': 'list_in_tuple'},
)
class TestOutputTypeCheck(unittest.TestCase):
    def test_output_type_check(self):
        class Model(chainer.Chain):
            def __init__(self, out_kind):
                super().__init__()
                self.out_kind = out_kind

            def __call__(self, x):
                if self.out_kind == 'array':
                    return x.array
                elif self.out_kind == 'array_in_tuple':
                    return x, x.array
                elif self.out_kind == 'list_in_tuple':
                    return ([x]),
                else:
                    assert self.out_kind == 'var'
                    return x

        model = Model(self.out_kind)
        x = np.ones((1, 3, 4, 5), dtype=np.float32)

        if self.out_kind == 'var':
            export(model, (x,))  # should be no error
        elif self.out_kind == 'array':
            with self.assertRaises(RuntimeError) as e:
                export(model, (x,))
            assert 'Unexpected output type'.find(e.exception.args[0])
        else:
            with self.assertRaises(ValueError) as e:
                export(model, (x,))
            assert 'must be Chainer Variable'.find(e.exception.args[0])


class TestUnusedLink(ONNXModelTest):

    # When some links are under init scope but not used on forwarding, params
    # of the links are not initialized. This means exporter cannot convert them
    # to ONNX's tensor because of lack of shape etc.

    def test_outputs(self):
        class MLP(chainer.Chain):
            def __init__(self, n_units, n_out):
                super(MLP, self).__init__()
                with self.init_scope():
                    self.l1 = L.Linear(None, n_units)
                    self.l2 = L.Linear(None, n_units)
                    self.l3 = L.Linear(None, n_out)

            def __call__(self, x):
                h1 = F.relu(self.l1(x))
                # Unused for some reason, then params are not initialized.
                # h2 = F.relu(self.l2(h1))
                return self.l3(h1)

        model = MLP(100, 10)
        x = np.random.rand(1, 768).astype(np.float32)

        with testing.assert_warns(UserWarning):
            self.expect(model, x)


@testing.parameterize(
    {
        'x_shape': (10, 3, 28, 28), 'shape_option': ('b', 3, 28, 28),
    },
    {
        'x_shape': (10, 3, 28, 28),
        'shape_option': [('b', 3, 28, 28)],
        'condition': 'var_list'
    },
    {
        'x_shape': [(10, 3, 28, 28), (8, 3, 28, 28)],
        'shape_option': [('b', 3, 28, 28), ('b', 3, 28, 28)],
        'condition': 'list_list'
    },
    {
        'x_shape': {'1': (10, 3, 28, 28), '2': (8, 3, 28, 28)},
        'shape_option': {'2': ('b', 3, 28, 28), '1': ('b', 3, 28, 28)},
        'condition': 'dict_dict'
    },
    {
        'x_shape': {'1': (10, 3, 28, 28), '2': (8, 3, 28, 28)},
        'shape_option': [('b', 3, 28, 28), ('b', 3, 28, 28)],
        'condition': 'dict_list'
    },
)
class TestCustomizedInputShape(ONNXModelTest):

    def test_output(self):
        class Model(chainer.Chain):
            def __init__(self):
                super().__init__()
                with self.init_scope():
                    self.l1 = L.Convolution2D(None, 16, 5, 1, 2)
                    self.l2 = L.Convolution2D(16, 8, 5, 1, 2)

            def forward(self, *xs, **kwxs):
                if kwxs:
                    h = F.vstack(list(kwxs.values()))
                elif len(xs) > 1:
                    h = F.vstack(xs)
                else:
                    h = xs[0]
                h2 = self.l1(h)
                h3 = F.relu(h2)
                h4 = self.l2(h3)
                return F.relu(h4)

        def check_input_shape(onnx_model, path):
            assert [v.type.tensor_type.shape.dim[0] == 'b' for
                    v in onnx_model.graph.input]
            assert [v.type.tensor_type.shape.dim[0] == 'b' for
                    v in onnx_model.graph.output]

        if isinstance(self.x_shape, tuple):
            xs = np.zeros(self.x_shape, dtype=np.float32)
        elif isinstance(self.x_shape, list):
            xs = tuple(
                np.zeros(shape, dtype=np.float32) for shape in self.x_shape)
        else:
            assert isinstance(self.x_shape, dict)
            xs = {k: np.zeros(shape, dtype=np.float32) for
                  k, shape in self.x_shape.items()}

        name = 'customized_input_shape'
        if hasattr(self, 'condition'):
            name += '_{}'.format(self.condition)

        self.expect(
            Model(), xs, name=name, input_shapes=self.shape_option,
            custom_model_test_func=check_input_shape)


@pytest.mark.parametrize('x_shape,shape_option', [
    ((10, 5), '?'),  # not tuple
    ((10, 5), ('?', 5, 5)),  # shape length error
    ((10, 5), [('?', 5), ('?', 5)]),  # not single
    ([(10, 5), (10, 5)], [('?', 5), ('?', 5), ('?', 5)]),  # list length error
    ([(10, 5), (10, 5)], [('?', 5), ('?', 5, 5)]),  # shape length error
    ({'a': (10, 5), 'b': (10, 5)}, {'a': ('?', 5), 'c': ('?', 5)}),  # NOQA not key found
    ({'a': (10, 5), 'b': (10, 5)}, [('?', 5), ('?', 5), ('?', 5)]),  # NOQA list length error
    ({'a': (10, 5), 'b': (10, 5)}, {'a': ('?', 5), 'b': ('?', 5, 5)}),  # NOQA not key found
])
def test_invalid_customized_input_shape(x_shape, shape_option):
    model = chainer.Sequential(F.relu)

    if isinstance(x_shape, tuple):
        xs = np.zeros(x_shape, dtype=np.float32)
    elif isinstance(x_shape, list):
        xs = tuple(
            np.zeros(shape, dtype=np.float32) for shape in x_shape)
    else:
        assert isinstance(x_shape, dict)
        xs = {k: np.zeros(shape, dtype=np.float32) for
              k, shape in x_shape.items()}

    with pytest.raises(ValueError):
        export(model, xs, input_shapes=shape_option)
