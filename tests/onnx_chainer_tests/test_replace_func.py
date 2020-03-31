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
from onnx_chainer_tests.helper import ONNXModelChecker
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


class TestReplaceWithOutputGrad(ONNXModelChecker):

    def get_model(self):
        class Model(chainer.Chain):

            def __init__(self):
                super().__init__()
                with self.init_scope():
                    self.l = L.Linear(None, 2)

            def half(self, xs, value=0.5):
                return xs * value

            def forward(self, xs):
                h = self.l(xs)
                h = self.half(h)
                return F.sum(chainer.as_variable(h))

        return Model()

    def test_grad_error(self):
        model = self.get_model()
        # this alternative function does not return chainer.Variable
        # backward propagation will fail
        model.half = fake_as_funcnode(
            lambda xs, value=0.5: xs.array * value, 'MulConstant')
        x = input_generator.increasing(2, 5)

        with pytest.raises(ValueError):
            self.expect(model, x, output_grad=True)

    def test_output(self, tmpdir):
        # first, make expected gradients to temp directory
        expected_result_path = str(tmpdir)

        model = self.get_model()
        x = input_generator.increasing(2, 5)
        export_testcase(model, x, expected_result_path, output_grad=True)

        data_set_name = 'test_data_set_0'
        expected_gradients = [os.path.join(
            expected_result_path, data_set_name, 'gradient_{}.pb').format(i)
            for i in range(2)]
        assert all([os.path.isfile(path) for path in expected_gradients])

        # model.half returns chainer.Variable and enabled backward
        # regardless using replacing
        model.half = fake_as_funcnode(model.half, 'MulConstant')
        x = input_generator.increasing(2, 5)

        def gradient_check(model, path):
            actual_gradients = [os.path.join(
                path, data_set_name, 'gradient_{}.pb').format(i)
                for i in range(2)]
            assert all([os.path.isfile(path) for path in actual_gradients])

            def load_tensor(path):
                tensor = onnx.load_tensor(path)
                return onnx.numpy_helper.to_array(tensor)

            for e_path, a_path in zip(expected_gradients, actual_gradients):
                expected = load_tensor(e_path)
                actual = load_tensor(a_path)
                np.testing.assert_allclose(expected, actual)

        self.expect(
            model, x, output_grad=True, custom_model_test_func=gradient_check)


class TestReplaceFuncBackward(ONNXModelTest):

    def _test_replace_func(self, fn, xs, set_grad=False):
        def make_list(v):
            if isinstance(v, (list, tuple)):
                return list(v)
            else:
                return [v]

        xvs = [x for x in xs if isinstance(x, chainer.Variable)]
        rfn = as_funcnode('fn')(fn)
        eys = make_list(fn(*xs))
        egxs = chainer.grad(eys, xvs, set_grad=set_grad)
        ays = make_list(rfn(*xs))
        agxs = chainer.grad(ays, xvs, set_grad=set_grad)
        assert len(eys) == len(ays)
        for ay, ey in zip(ays, eys):
            np.testing.assert_allclose(ay.array, ey.array)
        assert len(egxs) == len(agxs)
        for agx, egx in zip(agxs, egxs):
            if egx is None:
                assert egx is None
            else:
                np.testing.assert_allclose(agx.array, egx.array)

    def test_backward_simple(self):
        self._test_replace_func(lambda a, b: a * b,
                                [chainer.Variable(np.array(2.3)),
                                 chainer.Variable(np.array(4.2))])

    def test_backward_partially_differentiable(self):
        self._test_replace_func(lambda a, b: a * b.array,
                                [chainer.Variable(np.array(2.3)),
                                 chainer.Variable(np.array(4.2))])

    def test_backward_multi_outputs(self):
        self._test_replace_func(lambda a, b, c: (a * b, a / b, a * b * c),
                                [chainer.Variable(np.array(2.3)),
                                 chainer.Variable(np.array(4.2)),
                                 5])

    def test_backward_no_side_effect(self):
        a = chainer.Variable(np.array(2.3))
        b = chainer.Variable(np.array(4.2))
        x0 = a * b
        x1 = chainer.Variable(np.array(3.7))
        self._test_replace_func(lambda a, b: a * b, [x0, x1])
        # No side-effect to `grad`.
        assert x0.grad is None
        assert x1.grad is None
        assert a.grad is None
        assert b.grad is None
        # Gradient computation must stop at `x0` and `x1`.
        self._test_replace_func(lambda a, b: a * b, [x0, x1], set_grad=True)
        assert x0.grad is not None
        assert x1.grad is not None
        assert a.grad is None
        assert b.grad is None


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


def test_fake_as_funcnode_keep_structure(tmpdir):
    path = str(tmpdir)

    class Model(chainer.Chain):
        def __init__(self):
            super().__init__()

        def f(self, x):
            return {'a': (x, x+1), 'b': [x+2, x+3, x+4]}

        def __call__(self, x):
            ret = self.f(x)
            return ret['a'][0] + ret['b'][1]

    model = Model()
    x = input_generator.increasing(2, 3)

    with warnings.catch_warnings(record=True):
        model.f = fake_as_funcnode(model.f, 'xF')

    def f_converter(params):
        return onnx_helper.make_node(
            'xF', params.input_names, params.output_names),

    addon_converters = {'xF': f_converter}

    with testing.assert_warns(UserWarning):
        export_testcase(model, x, path, external_converters=addon_converters)

    model_filepath = os.path.join(path, 'model.onnx')
    assert os.path.isfile(model_filepath)

    onnx_model = onnx.load(model_filepath)
    node_names = [n.name for n in onnx_model.graph.node]
    assert len(node_names) == 2
    assert node_names[0] == 'xF_0'
    assert len(onnx_model.graph.node[0].output) == 5
    assert len(onnx_model.graph.output) == 1
