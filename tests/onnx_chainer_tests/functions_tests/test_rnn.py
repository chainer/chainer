import chainer
import chainer.functions as F
import chainer.links as L
from chainer import testing
import numpy as np

from onnx_chainer import onnx_helper
from onnx_chainer.testing import input_generator
from onnx_chainer_tests.helper import ONNXModelTest


@testing.parameterize(
    {'n_layers': 1, 'name': 'n_step_gru_1_layer'},
    {'n_layers': 2, 'name': 'n_step_gru_2_layer'},
)
class TestNStepGRU(ONNXModelTest):
    def test_output(self):
        n_layers = self.n_layers
        dropout_ratio = 0.0
        batch_size = 3
        input_size = 4
        hidden_size = 5
        seq_length = 6

        class Model(chainer.Chain):
            def __init__(self):
                super().__init__()

            def __call__(self, hx, ws1, ws2, ws3, bs, xs):
                ws = [F.separate(ws1) + F.separate(ws2)]
                if n_layers > 1:
                    ws.extend([F.separate(w) for w in F.separate(ws3)])
                bs = [F.separate(b) for b in F.separate(bs)]
                xs = F.separate(xs)
                hy, ys = F.n_step_gru(n_layers, dropout_ratio,
                                      hx, ws, bs, xs)
                return hy, F.stack(ys, axis=0)

        model = Model()

        hx = input_generator.increasing(n_layers, batch_size, hidden_size)
        ws1 = input_generator.increasing(3, hidden_size, input_size)
        ws2 = input_generator.increasing(3, hidden_size, hidden_size)
        ws3 = input_generator.increasing(
            n_layers - 1, 6, hidden_size, hidden_size)
        bs = input_generator.increasing(n_layers, 6, hidden_size)
        xs = input_generator.increasing(seq_length, batch_size, input_size)

        self.expect(model, (hx, ws1, ws2, ws3, bs, xs))


def convert_Permutate(params):
    gb = onnx_helper.GraphBuilder()
    # indices_name = params.context.get_name(func.indices)
    indices_name = params.context.add_const(params.func.indices,
                                            'indices')  # XXX
    if params.func.inv:
        empty = params.context.add_const(
            np.zeros(dtype=np.int64, shape=params.func.indices.shape), 'empty')
        r = params.context.add_const(
            np.arange(len(params.func.indices), dtype=np.int64),
            'range')
        op = 'ScatterElements' if params.opset_version == 11 else 'Scatter'
        indices_name = gb.op(op, [empty, indices_name, r])
    params.input_names.append(indices_name)
    gb.op_output_named('Gather', params.input_names, params.output_names,
                       axis=params.func.axis)
    return gb.nodes()


@testing.parameterize(
    {'n_layers': 1, 'name': 'TestNStepGRU_1_layer'},
    {'n_layers': 2, 'name': 'TestNStepGRU_2_layer'},
)
class TestNStepGRULink(ONNXModelTest):
    def test_output(self):
        n_layers = self.n_layers
        dropout_ratio = 0.0
        batch_size = 3
        input_size = 4
        hidden_size = 5
        seq_length = 6

        class Model(chainer.Chain):
            def __init__(self):
                super().__init__()
                with self.init_scope():
                    self.gru = L.NStepGRU(
                        n_layers, input_size, hidden_size, dropout_ratio)

            def __call__(self, *xs):
                hy, ys = self.gru(None, xs)
                return [hy] + ys

        model = Model()
        xs = [input_generator.increasing(seq_length, input_size)
              for i in range(batch_size)]

        # NOTE(msakai): Replace Permutate converter for avoiding error like:
        # ValidationError: Nodes in a graph must be topologically sorted, \
        # however input 'v330' of node:
        # input: "Permutate_0_const_empty" input: "v330" \
        # input: "Permutate_0_const_range" output: "Permutate_0_tmp_0" \
        # name: "Permutate_0_tmp_0" op_type: "Scatter"
        # is not output of any previous nodes.
        addon_converters = {
            'Permutate': convert_Permutate,
        }
        self.expect(model, xs, skip_opset_version=[7, 8],
                    external_converters=addon_converters)
