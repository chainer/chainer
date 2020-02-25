import chainer

from onnx_chainer.functions.opset_version import support
from onnx_chainer import onnx_helper


@support((1, 6, 7))
def convert_n_step_gru(func, opset_version, input_names, output_names,
                       context):
    n_layers, dropout_ratio, hx, ws, bs, xs = func.args
    assert n_layers >= 1
    hidden_size = hx.shape[2]

    gb = onnx_helper.GraphBuilder()

    hx_name = input_names[0]
    offset = 1
    ws_names = [[input_names[offset + i * 6 + j] for j in range(6)]
                for i in range(n_layers)]
    offset += 6 * n_layers
    bs_names = [[input_names[offset + i * 6 + j] for j in range(6)]
                for i in range(n_layers)]
    offset += 6 * n_layers
    xs_names = input_names[offset:]

    split_outs = gb.op('Split', [hx_name], num_outputs=n_layers, axis=0)
    if n_layers == 1:
        split_outs = [split_outs]
    # Removing layer dimention and adding num_directions cancels each other
    hx_names = split_outs

    hy_name, ys_name_list = \
        func.reconstruct_return_value(output_names)

    y_name = None
    hy_names = []

    for layer in range(n_layers):
        if layer == 0:
            # X; shape: (seq_length, batch_size, input_size)
            x_name = gb.op(
                'Concat',
                [gb.op('Unsqueeze', [name], axes=[0]) for name in xs_names],
                axis=0)
        else:
            if opset_version >= 7:
                x_name = gb.op('Dropout', [y_name], ratio=dropout_ratio)
            elif opset_version >= 6:
                x_name = gb.op('Dropout', [y_name], ratio=dropout_ratio,
                               is_test=0 if chainer.config.train else 1)
            else:
                x_name = gb.op('Dropout', [y_name], ratio=dropout_ratio,
                               is_test=0 if chainer.config.train else 1,
                               consumed_inputs=[1])

            # remove num_directions dimention
            x_name = gb.op('Squeeze', [x_name], axes=[1])

        w = ws_names[layer]
        b = bs_names[layer]

        # W[zrh]; shape: (num_directions, 3*hidden_size, input_size)
        w_name = gb.op(
            'Unsqueeze',
            [gb.op('Concat', [w[1], w[0], w[2]], axis=0)],
            axes=[0])
        # R[zrh]; shape: (num_directions, 3*hidden_size, input_size)
        r_name = gb.op(
            'Unsqueeze',
            [gb.op('Concat', [w[4], w[3], w[5]], axis=0)],
            axes=[0])
        # Wb[zrh], Rb[zrh]; shape: (num_directions, 6*hidden_size)
        b_name = gb.op(
            'Unsqueeze',
            [gb.op('Concat', [b[1], b[0], b[2], b[4], b[3], b[5]], axis=0)],
            axes=[0])

        # Y; shape: (seq_length, num_directions, batch_size, hidden_size)
        # Y_h; shape: (num_directions, batch_size, hidden_size)
        y_name, hy_name_ = gb.op(
            'GRU',
            (x_name, w_name, r_name, b_name, "", hx_names[layer]),
            hidden_size=hidden_size,
            linear_before_reset=1,
            num_outputs=2)
        hy_names.append(hy_name_)

    split_outs = gb.op(
        'Split',
        # remove num_directions dimention
        [gb.op('Squeeze', [y_name], axes=[1])],
        num_outputs=len(ys_name_list), axis=0)
    if len(ys_name_list) == 1:
        split_outs = [split_outs]
    for i, node_name in enumerate(split_outs):
        # remove seq_length dimention
        gb.op_output_named('Squeeze', [node_name], [ys_name_list[i]], axes=[0])

    # Removal of num_directions and new dimention for concatenation cancel
    # each other.
    gb.op_output_named('Concat', hy_names, [hy_name], axis=0)

    return gb.nodes()
