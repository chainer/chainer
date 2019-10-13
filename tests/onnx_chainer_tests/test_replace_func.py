import os
import warnings

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import testing
import numpy as np
import onnx
import pytest

from onnx_chainer import export
from onnx_chainer import export_testcase
from onnx_chainer import onnx_helper
from onnx_chainer.replace_func import as_funcnode
from onnx_chainer.replace_func import fake_as_funcnode
from onnx_chainer.testing import input_generator
from onnx_chainer_tests.helper import ONNXModelTest


def test_fake_as_funcnode_without_replace():

    class Model(chainer.Chain):
        def _init__(self):
            super().__init__()

        def add(self, xs, value=0.01):
            return xs.array + value

        def __call__(self, xs):
            return F.sigmoid(self.add(xs))

    model = Model()
    x = input_generator.increasing(3, 4)

    onnx_model = export(model, x)
    sigmoid_nodes = [
        node for node in onnx_model.graph.node if node.op_type == 'Sigmoid']
    assert len(sigmoid_nodes) == 1
    # sigmoid node should be expected to connect with input
    # but the connection is cut because `add` method takes array.
    assert not sigmoid_nodes[0].input[0] == 'Input_0'


class TestReplaceNumpyFullToConstantOfShape(ONNXModelTest):
    # This test case is a real-world example, to handle np.full
    def test_output(self):
        class Model(chainer.Chain):
            def __init__(self, value):
                super().__init__()
                self.value = value

            @as_funcnode('NumpyFull')
            def full(self, xs, value=0):
                # not support `def full(self, xs_shape, value=0)`
                # wrapped function node cannot handle shape directly yet.
                return np.full(xs.array.shape, value, dtype=np.float32)

            def __call__(self, xs):
                return F.sigmoid(self.full(xs, value=self.value))

        model = Model(value=5)
        x = input_generator.increasing(2, 3, 4)

        def numpy_full_converter(params):
            gb = onnx_helper.GraphBuilder()
            output = gb.op('Shape', params.input_names)
            value = onnx.helper.make_tensor(
                'value', onnx.TensorProto.FLOAT, [1], [params.func.value])
            gb.op_output_named(
                'ConstantOfShape', [output], params.output_names, value=value)
            return gb.nodes()

        addon_converters = {'NumpyFull': numpy_full_converter}

        self.expect(
            model, x, skip_opset_version=[7, 8],
            external_converters=addon_converters)


class TestReplaceWithOutputGrad(ONNXModelTest):

    def test_output(self):
        class Model(chainer.Chain):

            def __init__(self):
                super().__init__()
                with self.init_scope():
                    self.l = L.Linear(None, 2)

            @as_funcnode('MulConstant')
            def half(self, xs, value=0.5):
                return xs * value

            def forward(self, xs):
                h = self.l(xs)
                h = self.half(h)
                return F.sum(chainer.as_variable(h))

        model = Model()
        x = input_generator.increasing(2, 5)

        def gradient_check(m, test_path):
            test_data_set_path = os.path.join(test_path, 'test_data_set_0')

            gradient0_path = os.path.join(test_data_set_path, 'gradient_0.pb')
            onnx_tensor = onnx.load_tensor(gradient0_path)
            actual = onnx.numpy_helper.to_array(onnx_tensor)
            expect = np.array([[-1.25, -0.75, -0.25, 0.25, 0.75],
                               [-1.25, -0.75, -0.25, 0.25, 0.75]],
                              dtype=np.float32)
            np.testing.assert_allclose(actual, expect)

            gradient1_path = os.path.join(test_data_set_path, 'gradient_1.pb')
            onnx_tensor = onnx.load_tensor(gradient1_path)
            actual = onnx.numpy_helper.to_array(onnx_tensor)
            expect = np.array([1., 1.], dtype=np.float32)
            np.testing.assert_allclose(actual, expect)

        self.expect(
            model, x, output_grad=True, custom_model_test_func=gradient_check)


@testing.parameterize(
    {'func_kind': 'list', 'in_shape': (2, 3, 4), 'op_type': 'Add'},
    {'func_kind': 'list_kwargs', 'in_shape': (2, 3, 4), 'op_type': 'Add'},
    {'func_kind': 'var_with_deco', 'in_shape': (3, 4),
     'op_type': 'AddConstant'},
    {'func_kind': 'var_kwargs', 'in_shape': (3, 4), 'op_type': 'AddConstant'},
    {'func_kind': 'var', 'in_shape': (3, 4), 'op_type': 'AddConstant'},
)
class TestReplaceFunc(ONNXModelTest):

    def get_model(self, target_func, input_converter):
        class Model(chainer.Chain):
            def __init__(self, target_func, input_converter):
                super().__init__()
                self.input_converter = input_converter
                self.fn = target_func

            def __call__(self, xs):
                args, kwargs = self.input_converter(xs)
                h = self.fn(*args, **kwargs)
                return F.sigmoid(h)

        return Model(target_func, input_converter)

    def test_output(self):
        attr = None
        is_deco = False
        if self.func_kind == 'list':
            def input_converter(xs):
                return ([xs[0], xs[1]],), {}

            def target_func(xs):
                return xs[0].array + xs[1].array

        elif self.func_kind == 'list_kwargs':
            def input_converter(xs):
                return (), {'xs': [xs[0], xs[1]]}

            def target_func(xs=None):
                assert xs is not None
                return xs[0].array + xs[1].array

        elif self.func_kind == 'var_with_deco':
            def input_converter(xs):
                return (xs,), {}

            @as_funcnode('AddConstant', rename_attributes=[('b', 'value')])
            def target_func(x, b=0.01):
                return x.array + b

            is_deco = True

        elif self.func_kind == 'var_kwargs':
            def input_converter(xs):
                return (), {'x': xs, 'value': 0.02}

            def target_func(x=None, value=0.01):
                assert x is not None
                return x.array + value

        else:
            assert self.func_kind == 'var'

            def input_converter(xs):
                return (xs, 0.01), {}

            def target_func(x, value):
                return x.array + value

            attr = [(1, 'value')]

        model = self.get_model(target_func, input_converter)
        x = input_generator.increasing(*self.in_shape)

        if not is_deco:
            model.fn = fake_as_funcnode(
                model.fn, self.op_type, rename_attributes=attr)

        name = 'replace_func_' + self.func_kind
        self.expect(model, x, name=name)


@pytest.mark.parametrize('return_type', ['list', 'dict'])
def test_replace_func_collection_return(tmpdir, return_type):
    path = str(tmpdir)

    class Model(chainer.Chain):
        def __init__(self, return_type):
            super().__init__()
            self.return_type = return_type

        def tiled_array(self, xs, n=5):
            if self.return_type == 'list':
                return [xs.array * i for i in range(1, 1+n)]
            else:
                assert self.return_type == 'dict'
                return {str(i): xs.array * i for i in range(1, 1+n)}

        def __call__(self, xs):
            return self.tiled_array(xs)

    model = Model(return_type)
    x = input_generator.increasing(1, 5)

    with warnings.catch_warnings(record=True):
        model.tiled_array = fake_as_funcnode(model.tiled_array, 'xTiledArray')

    def tiled_array_converter(params):
        return onnx_helper.make_node(
            'xTiledArray', params.input_names, params.output_names),

    addon_converters = {'xTiledArray': tiled_array_converter}

    with testing.assert_warns(UserWarning):
        export_testcase(model, x, path, external_converters=addon_converters)

    model_filepath = os.path.join(path, 'model.onnx')
    assert os.path.isfile(model_filepath)

    onnx_model = onnx.load(model_filepath)
    node_names = [n.name for n in onnx_model.graph.node]
    assert len(node_names) == 1
    assert node_names[0] == 'xTiledArray_0'
    output_names = [n.name for n in onnx_model.graph.output]
    assert len(output_names) == 5
    for i, name in enumerate(output_names):
        assert name == 'xTiledArray_0_{:d}'.format(i)
