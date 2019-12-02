import chainer
import chainer.functions as F
import chainer.links as L
from chainer import testing

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
        self.expect(model, xs, skip_opset_version=[7, 8])
