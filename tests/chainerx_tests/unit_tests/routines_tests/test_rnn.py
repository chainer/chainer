import chainer

import chainerx

from chainerx_tests import array_utils
from chainerx_tests import dtype_utils
from chainerx_tests import op_utils

n_step_lstm_dtypes_valid = dtype_utils._permutate_dtype_mapping([
    # Floats.
    (('float16', ), ()),
    (('float32', ), ()),
    (('float64', ), ()),
])


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product([
        chainer.testing.from_pytest_parameterize(
            'n_layers,hidden_size,input_size,batches', [
                (2, 2, 1, (1, 1, 1)),
                (2, 2, 3, (3, 2, 1)),
                (3, 8, 4, (4, 2, 1)),
                (4, 12, 4, (4, 3, 2)),

            ]),
        chainer.testing.from_pytest_parameterize(
            'in_dtypes, out_dtype', n_step_lstm_dtypes_valid)
    ])
))
class TestNStepLstm(op_utils.ChainerOpTest):

    def setup(self):
        self.check_forward_options.update({
            'rtol': 1e-2, 'atol': 1e-2})
        self.check_backward_options.update({
            'rtol': 1e-2, 'atol': 1e-2})
        self.check_double_backward_options.update({
            'rtol': 5e-3, 'atol': 5e-2})
        if self.in_dtypes[0] == 'float16':
            self.check_forward_options.update({
                'rtol': 1e-1, 'atol': 1e-1})
            self.check_backward_options.update({
                'rtol': 1e-1, 'atol': 1e-1})
            self.check_double_backward_options.update({
                'rtol': 1e-1, 'atol': 1e-1})
        device = chainerx.get_default_device()
        if device.backend.name == 'cuda':
            if self.in_dtypes[0] != 'float32':
                self.skip_backward_test = True
            self.skip_double_backward_test = True

    def generate_inputs(self):
        h_shape = (self.n_layers, self.batches[0], self.hidden_size)
        dtype = self.in_dtypes[0]

        h = array_utils.uniform(h_shape, dtype)
        c = array_utils.uniform(h_shape, dtype)
        in_size = self.input_size
        out_size = self.hidden_size
        xs = [array_utils.uniform((self.batches[b], in_size), dtype)
              for b in range(len(self.batches))]

        def w_in(i, j):
            return in_size if i == 0 and j < 4 else out_size

        inputs = []
        inputs.append(h)
        inputs.append(c)
        for i in range(len(self.batches)):
            inputs.append(xs[i])

        for n in range(self.n_layers):
            for i in range(8):
                inputs.append(array_utils.uniform(
                    (out_size, w_in(n, i)), dtype))
            for i in range(8):
                inputs.append(array_utils.uniform((out_size,), dtype))
        return tuple(inputs)

    def process_input(self, inputs):
        h = inputs[0]
        c = inputs[1]
        xs = inputs[2:2 + len(self.batches)]
        ws = []
        bs = []
        index = 2 + len(self.batches)
        for n in range(self.n_layers):
            ws.append(inputs[index: index + 8])
            bs.append(inputs[index + 8: index + 16])
            index += 16
        return h, c, ws, bs, xs

    def forward_chainerx(self, inputs):
        h, c, ws, bs, xs = self.process_input(inputs)
        out = chainerx.n_step_lstm(self.n_layers, h, c, ws, bs, xs)
        rets = []
        rets.append(out[0])
        rets.append(out[1])
        for i in range(len(out[2])):
            rets.append(out[2][i])
        return tuple(rets)

    def forward_chainer(self, inputs):
        h, c, ws, bs, xs = self.process_input(inputs)
        out = chainer.functions.n_step_lstm(
            self.n_layers, 0.0, h, c, ws, bs, xs)
        rets = []
        rets.append(out[0])
        rets.append(out[1])
        for i in range(len(out[2])):
            rets.append(out[2][i])

        return tuple(rets)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product([
        chainer.testing.from_pytest_parameterize(
            'n_layers,hidden_size,input_size,batches', [
                (1, 2, 1, (1, 1, 1)),
                (2, 6, 8, (4, 2, 2)),
                (3, 8, 4, (4, 2, 1)),
                (4, 12, 4, (4, 3, 2)),

            ]),
        chainer.testing.from_pytest_parameterize(
            'in_dtypes,out_dtype', n_step_lstm_dtypes_valid)
    ])
))
class TestNStepBiLstm(op_utils.ChainerOpTest):

    def setup(self):

        self.check_forward_options.update({
            'rtol': 1e-2, 'atol': 1e-2})
        self.check_backward_options.update({
            'rtol': 1e-2, 'atol': 1e-2})
        self.check_double_backward_options.update({
            'rtol': 5e-3, 'atol': 5e-2})
        if self.in_dtypes[0] == 'float16':
            self.check_forward_options.update({
                'rtol': 1e-1, 'atol': 1e-1})
            self.check_backward_options.update({
                'rtol': 1e-1, 'atol': 1e-1})
            self.check_double_backward_options.update({
                'rtol': 1e-1, 'atol': 1e-1})
        device = chainerx.get_default_device()
        if device.backend.name == 'cuda':
            if self.in_dtypes[0] != 'float32':
                self.skip_backward_test = True
            self.skip_double_backward_test = True

    def generate_inputs(self):
        h_shape = (self.n_layers * 2, self.batches[0], self.hidden_size)
        dtype = self.in_dtypes[0]

        h = array_utils.uniform(h_shape, dtype)
        c = array_utils.uniform(h_shape, dtype)
        in_size = self.input_size
        out_size = self.hidden_size
        xs = [array_utils.uniform((self.batches[b], in_size), dtype)
              for b in range(len(self.batches))]

        def w_in(i, j):
            if i == 0 and j < 4:
                return in_size
            elif i > 0 and j < 4:
                return out_size * 2
            else:
                return out_size

        inputs = []
        inputs.append(h)
        inputs.append(c)
        for i in range(len(self.batches)):
            inputs.append(xs[i])

        for n in range(self.n_layers):
            for direction in (0, 1):
                for i in range(8):
                    inputs.append(array_utils.uniform(
                        (out_size, w_in(n, i)), dtype))
                for i in range(8):
                    inputs.append(array_utils.uniform((out_size,), dtype))
        return tuple(inputs)

    def process_input(self, inputs):
        h = inputs[0]
        c = inputs[1]
        xs = inputs[2:2 + len(self.batches)]
        ws = []
        bs = []
        index = 2 + len(self.batches)
        for n in range(self.n_layers):
            ws.append(inputs[index: index + 8])
            bs.append(inputs[index + 8: index + 16])
            ws.append(inputs[index + 16: index + 24])
            bs.append(inputs[index + 24: index + 32])
            index += 32
        return h, c, ws, bs, xs

    def forward_chainerx(self, inputs):
        h, c, ws, bs, xs = self.process_input(inputs)
        out = chainerx.n_step_bilstm(self.n_layers, h, c, ws, bs, xs)
        rets = []
        rets.append(out[0])
        rets.append(out[1])
        for i in range(len(out[2])):
            rets.append(out[2][i])
        return tuple(rets)

    def forward_chainer(self, inputs):
        h, c, ws, bs, xs = self.process_input(inputs)
        out = chainer.functions.n_step_bilstm(
            self.n_layers, 0.0, h, c, ws, bs, xs)
        rets = []
        rets.append(out[0])
        rets.append(out[1])
        for i in range(len(out[2])):
            rets.append(out[2][i])

        return tuple(rets)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product([
        chainer.testing.from_pytest_parameterize(
            'n_layers,hidden_size,input_size,batches', [
                (2, 2, 1, (1, 1, 1)),
                (2, 2, 3, (3, 2, 1)),
                (3, 8, 4, (4, 2, 1)),
                (4, 6, 4, (4, 3, 2)),

            ]),
        chainer.testing.from_pytest_parameterize(
            'in_dtypes, out_dtype', n_step_lstm_dtypes_valid)
    ])
))
@op_utils.fix_random()  # This test is unstable.
class TestNStepGru(op_utils.ChainerOpTest):

    def setup(self):
        self.check_forward_options.update({
            'rtol': 1e-2, 'atol': 1e-2})
        self.check_backward_options.update({
            'rtol': 1e-2, 'atol': 1e-2})
        self.check_double_backward_options.update({
            'rtol': 5e-3, 'atol': 5e-2})
        if self.in_dtypes[0] == 'float16':
            self.check_forward_options.update({
                'rtol': 1e-1, 'atol': 1e-1})
            self.check_backward_options.update({
                'rtol': 1e-1, 'atol': 1e-1})
            self.check_double_backward_options.update({
                'rtol': 1e-1, 'atol': 1e-1})
        device = chainerx.get_default_device()
        if device.backend.name == 'cuda':
            if self.in_dtypes[0] != 'float32':

                self.skip_backward_test = True
            self.skip_double_backward_test = True

    def generate_inputs(self):
        h_shape = (self.n_layers, self.batches[0], self.hidden_size)
        dtype = self.in_dtypes[0]

        h = array_utils.uniform(h_shape, dtype)

        in_size = self.input_size
        out_size = self.hidden_size
        xs = [array_utils.uniform((self.batches[b], in_size), dtype)
              for b in range(len(self.batches))]

        def w_in(i, j):
            return in_size if i == 0 and j < 3 else out_size

        inputs = []
        inputs.append(h)
        for i in range(len(self.batches)):
            inputs.append(xs[i])

        for n in range(self.n_layers):
            for i in range(6):
                inputs.append(array_utils.uniform(
                    (out_size, w_in(n, i)), dtype))
            for i in range(6):
                inputs.append(array_utils.uniform((out_size,), dtype))
        return tuple(inputs)

    def process_input(self, inputs):
        h = inputs[0]
        xs = inputs[1:1 + len(self.batches)]
        ws = []
        bs = []
        index = 1 + len(self.batches)
        for n in range(self.n_layers):
            ws.append(inputs[index: index + 6])
            bs.append(inputs[index + 6: index + 12])
            index += 12
        return h, ws, bs, xs

    def forward_chainerx(self, inputs):
        h, ws, bs, xs = self.process_input(inputs)
        out = chainerx.n_step_gru(self.n_layers, h, ws, bs, xs)
        rets = []
        rets.append(out[0])
        for i in range(len(out[1])):
            rets.append(out[1][i])
        return tuple(rets)

    def forward_chainer(self, inputs):
        h, ws, bs, xs = self.process_input(inputs)
        out = chainer.functions.n_step_gru(
            self.n_layers, 0.0, h, ws, bs, xs)
        rets = []
        rets.append(out[0])
        for i in range(len(out[1])):
            rets.append(out[1][i])

        return tuple(rets)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product([
        chainer.testing.from_pytest_parameterize(
            'n_layers,hidden_size,input_size,batches', [
                (2, 2, 1, (1, 1, 1)),
                (2, 2, 3, (3, 2, 1)),
                (3, 4, 4, (4, 2, 1)),
                (4, 5, 4, (4, 3, 2)),

            ]),
        chainer.testing.from_pytest_parameterize(
            'in_dtypes,out_dtype', n_step_lstm_dtypes_valid)
    ])
))
class TestNStepBiGRU(op_utils.ChainerOpTest):

    def setup(self):

        self.check_forward_options.update({
            'rtol': 1e-2, 'atol': 1e-2})
        self.check_backward_options.update({
            'rtol': 1e-2, 'atol': 1e-2})
        self.check_double_backward_options.update({
            'rtol': 5e-2, 'atol': 5e-2})
        if self.in_dtypes[0] == 'float16':
            self.check_forward_options.update({
                'rtol': 1e-1, 'atol': 1e-1})
            self.check_backward_options.update({
                'rtol': 1e-1, 'atol': 1e-1})
            self.check_double_backward_options.update({
                'rtol': 1e-1, 'atol': 1e-1})
        device = chainerx.get_default_device()
        if device.backend.name == 'cuda':
            if self.in_dtypes[0] != 'float32':
                self.skip_backward_test = True
            self.skip_double_backward_test = True

    def generate_inputs(self):
        h_shape = (self.n_layers * 2, self.batches[0], self.hidden_size)
        dtype = self.in_dtypes[0]

        h = array_utils.uniform(h_shape, dtype)
        in_size = self.input_size
        out_size = self.hidden_size
        xs = [array_utils.uniform((self.batches[b], in_size), dtype)
              for b in range(len(self.batches))]

        def w_in(i, j):
            if i == 0 and j < 3:
                return in_size
            elif i > 0 and j < 3:
                return out_size * 2
            else:
                return out_size

        inputs = []
        inputs.append(h)
        for i in range(len(self.batches)):
            inputs.append(xs[i])

        for n in range(self.n_layers):
            for direction in (0, 1):
                for i in range(6):
                    inputs.append(array_utils.uniform(
                        (out_size, w_in(n, i)), dtype))
                for i in range(6):
                    inputs.append(array_utils.uniform((out_size,), dtype))
        return tuple(inputs)

    def process_input(self, inputs):
        h = inputs[0]
        xs = inputs[1:1 + len(self.batches)]
        ws = []
        bs = []
        index = 1 + len(self.batches)
        for n in range(self.n_layers):
            ws.append(inputs[index: index + 6])
            bs.append(inputs[index + 6: index + 12])
            ws.append(inputs[index + 12: index + 18])
            bs.append(inputs[index + 18: index + 24])
            index += 24
        return h, ws, bs, xs

    def forward_chainerx(self, inputs):
        h, ws, bs, xs = self.process_input(inputs)
        out = chainerx.n_step_bigru(self.n_layers, h, ws, bs, xs)
        rets = []
        rets.append(out[0])
        for i in range(len(out[1])):
            rets.append(out[1][i])
        return tuple(rets)

    def forward_chainer(self, inputs):
        h, ws, bs, xs = self.process_input(inputs)
        out = chainer.functions.n_step_bigru(
            self.n_layers, 0.0, h, ws, bs, xs)
        rets = []
        rets.append(out[0])
        for i in range(len(out[1])):
            rets.append(out[1][i])

        return tuple(rets)


@op_utils.op_test(['native:0', 'cuda:0'])
# ReLU activation is unstable around 0 but can seemingly not be dodged
# automatically.
@op_utils.fix_random()
@chainer.testing.parameterize(*(
    chainer.testing.product([
        chainer.testing.from_pytest_parameterize(
            'n_layers,hidden_size,input_size,batches,activation', [
                (2, 2, 1, (1, 1, 1), "tanh"),
                (2, 2, 1, (1, 1, 1), "relu"),
                (2, 2, 3, (3, 2, 1), "tanh"),
                (2, 2, 3, (3, 2, 1), "relu"),
                (3, 4, 4, (4, 2, 1), "tanh"),
                (3, 4, 4, (4, 2, 1), "relu"),
                (4, 5, 4, (4, 3, 2), "tanh"),
                (4, 5, 4, (4, 3, 2), "relu"),

            ]),
        chainer.testing.from_pytest_parameterize(
            'in_dtypes, out_dtype', n_step_lstm_dtypes_valid)
    ])
))
class TestNStepRNN(op_utils.ChainerOpTest):
    check_numpy_strides_compliance = False
    dodge_nondifferentiable = True

    def setup(self):
        self.check_forward_options.update({
            'rtol': 1e-2, 'atol': 1e-2})
        self.check_backward_options.update({
            'rtol': 1e-2, 'atol': 1e-2})
        self.check_double_backward_options.update({
            'rtol': 5e-2, 'atol': 5e-2})
        if self.in_dtypes[0] == 'float16':
            self.check_forward_options.update({
                'rtol': 1e-1, 'atol': 1e-1})
            self.check_backward_options.update({
                'rtol': 1e-1, 'atol': 1e-1})
            self.check_double_backward_options.update({
                'rtol': 1e-1, 'atol': 1e-1})
        device = chainerx.get_default_device()
        if device.backend.name == 'cuda':
            if self.in_dtypes[0] != 'float32':
                self.skip_forward_test = True
                self.skip_backward_test = True
            self.skip_double_backward_test = True

    def generate_inputs(self):
        h_shape = (self.n_layers, self.batches[0], self.hidden_size)
        dtype = self.in_dtypes[0]

        h = array_utils.uniform(h_shape, dtype)

        in_size = self.input_size
        out_size = self.hidden_size
        xs = [array_utils.uniform((self.batches[b], in_size), dtype)
              for b in range(len(self.batches))]

        def w_in(i, j):
            return in_size if i == 0 and j < 1 else out_size

        inputs = []
        inputs.append(h)
        for i in range(len(self.batches)):
            inputs.append(xs[i])

        for n in range(self.n_layers):
            for i in range(2):
                inputs.append(array_utils.uniform(
                    (out_size, w_in(n, i)), dtype))
            for i in range(2):
                inputs.append(array_utils.uniform((out_size,), dtype))
        return tuple(inputs)

    def process_input(self, inputs):
        h = inputs[0]
        xs = inputs[1:1 + len(self.batches)]
        ws = []
        bs = []
        index = 1 + len(self.batches)
        for n in range(self.n_layers):
            ws.append(inputs[index: index + 2])
            bs.append(inputs[index + 2: index + 4])
            index += 4
        return h, ws, bs, xs

    def forward_chainerx(self, inputs):
        h, ws, bs, xs = self.process_input(inputs)
        out = chainerx.n_step_rnn(
            self.n_layers, h, ws, bs, xs, self.activation)
        rets = []
        rets.append(out[0])
        for i in range(len(out[1])):
            rets.append(out[1][i])
        return tuple(rets)

    def forward_chainer(self, inputs):
        h, ws, bs, xs = self.process_input(inputs)
        out = chainer.functions.n_step_rnn(
            self.n_layers, 0.0, h, ws, bs, xs, self.activation)
        rets = []
        rets.append(out[0])
        for i in range(len(out[1])):
            rets.append(out[1][i])

        return tuple(rets)


@op_utils.op_test(['native:0', 'cuda:0'])
# ReLU activation is unstable around 0 but can seemingly not be dodged
# automatically.
@op_utils.fix_random()
@chainer.testing.parameterize(*(
    chainer.testing.product([
        chainer.testing.from_pytest_parameterize(
            'n_layers,hidden_size,input_size,batches,activation', [
                (2, 2, 1, (1, 1, 1), "tanh"),
                (2, 2, 1, (1, 1, 1), "relu"),
                (2, 2, 3, (3, 2, 1), "tanh"),
                (2, 2, 3, (3, 2, 1), "relu"),
                (3, 4, 4, (4, 2, 1), "tanh"),
                (3, 4, 4, (4, 2, 1), "relu"),

            ]),
        chainer.testing.from_pytest_parameterize(
            'in_dtypes,out_dtype', n_step_lstm_dtypes_valid)
    ])
))
class TestNStepBiRNN(op_utils.ChainerOpTest):
    check_numpy_strides_compliance = False
    dodge_nondifferentiable = True

    def setup(self):

        self.check_forward_options.update({
            'rtol': 1e-2, 'atol': 1e-2})
        self.check_backward_options.update({
            'rtol': 1e-2, 'atol': 1e-2})
        self.check_double_backward_options.update({
            'rtol': 5e-2, 'atol': 5e-2})
        if self.in_dtypes[0] == 'float16':
            self.check_forward_options.update({
                'rtol': 1e-1, 'atol': 1e-1})
            self.check_backward_options.update({
                'rtol': 1e-1, 'atol': 1e-1})
            self.check_double_backward_options.update({
                'rtol': 1e-1, 'atol': 1e-1})
        device = chainerx.get_default_device()
        if device.backend.name == 'cuda':
            if self.in_dtypes[0] != 'float32':
                self.skip_forward_test = True
                self.skip_backward_test = True
            self.skip_double_backward_test = True

    def generate_inputs(self):
        h_shape = (self.n_layers * 2, self.batches[0], self.hidden_size)
        dtype = self.in_dtypes[0]
        low = -1.0
        high = 1.0
        if dtype == 'float16':
            low = -0.5
            high = 0.5

        h = array_utils.uniform(h_shape, dtype)
        in_size = self.input_size
        out_size = self.hidden_size
        xs = [array_utils.uniform((self.batches[b], in_size),
                                  dtype, low=low, high=high)
              for b in range(len(self.batches))]

        def w_in(i, j):
            if i == 0 and j < 1:
                return in_size
            elif i > 0 and j < 1:
                return out_size * 2
            else:
                return out_size

        inputs = []
        inputs.append(h)
        for i in range(len(self.batches)):
            inputs.append(xs[i])

        for n in range(self.n_layers):
            for direction in (0, 1):
                for i in range(2):
                    inputs.append(array_utils.uniform(
                        (out_size, w_in(n, i)), dtype, low=low, high=high))
                for i in range(2):
                    inputs.append(array_utils.uniform(
                        (out_size,), dtype, low=low, high=high))
        return tuple(inputs)

    def process_input(self, inputs):
        h = inputs[0]
        xs = inputs[1:1 + len(self.batches)]
        ws = []
        bs = []
        index = 1 + len(self.batches)
        for n in range(self.n_layers):
            ws.append(inputs[index: index + 2])
            bs.append(inputs[index + 2: index + 4])
            ws.append(inputs[index + 4: index + 6])
            bs.append(inputs[index + 6: index + 8])
            index += 8
        return h, ws, bs, xs

    def forward_chainerx(self, inputs):
        h, ws, bs, xs = self.process_input(inputs)
        out = chainerx.n_step_birnn(
            self.n_layers, h, ws, bs, xs, self.activation)
        rets = []
        rets.append(out[0])
        for i in range(len(out[1])):
            rets.append(out[1][i])
        return tuple(rets)

    def forward_chainer(self, inputs):
        h, ws, bs, xs = self.process_input(inputs)
        out = chainer.functions.n_step_birnn(
            self.n_layers, 0.0, h, ws, bs, xs, self.activation)
        rets = []
        rets.append(out[0])
        for i in range(len(out[1])):
            rets.append(out[1][i])

        return tuple(rets)
