import sys

import chainer
import numpy as np

from onnx_chainer.functions.array import get_slice_node
from onnx_chainer.functions.opset_version import support
from onnx_chainer import onnx_helper


@support((1, 6, 7))
def convert_BatchNormalization(
        func, opset_version, input_names, output_names, context):
    is_fixed_bn = len(func.inputs) > 3

    # NOTE: even if `use_beta=False` or `use_gamma=False`, beta or gamma
    # are set in inputs by RetainHook,
    beta_param = func.inputs[2].get_variable_or_none()
    gamma_param = func.inputs[1].get_variable_or_none()
    namedlink = context.get_link(beta_param) or context.get_link(gamma_param)

    if namedlink is not None:
        prefix, link = namedlink
        if is_fixed_bn:
            mean = link.avg_mean
            var = link.avg_var
        else:
            # on train mode, avg_mean would be updated, so make them from x
            x = func.inputs[0].get_variable().array
            mean = x.mean(axis=func.axis)
            var = x.var(axis=func.axis)
    else:
        prefix = None
        if is_fixed_bn:
            mean = func.inputs[3].get_variable().array
            var = func.inputs[4].get_variable().array
        else:
            x = func.inputs[0].get_variable().array
            mean = x.mean(axis=func.axis)
            var = x.var(axis=func.axis)

    def add_param(v, suffix):
        if prefix is None:
            return context.add_param(v, suffix)
        else:
            return context.add_param(
                v, '{}_{}'.format(prefix, suffix), use_original_name=True)

    if is_fixed_bn:
        if context.implicit_inputs.pop(input_names[3], None) is not None:
            mean_name = add_param(mean, 'avg_mean')
            input_names[3] = mean_name
        if context.implicit_inputs.pop(input_names[4], None) is not None:
            var_name = add_param(var, 'avg_var')
            input_names[4] = var_name
    else:
        maen_name = add_param(mean, 'avg_mean')
        var_name = add_param(var, 'avg_var')
        input_names.extend([maen_name, var_name])

    momentum = getattr(func, 'decay', 0.)

    # TODO(disktnk): On definition of ONNX's BatchNormalization operator,
    # outputs one required output and four optional outputs. This converter
    # must make 5 values for output and return them.

    if opset_version == 1:
        return onnx_helper.make_node(
            'BatchNormalization', input_names, output_names,
            epsilon=func.eps,
            momentum=momentum,
            is_test=not chainer.config.train,
            consumed_inputs=[False, False, False, True, True],
        ),
    elif opset_version == 6:
        return onnx_helper.make_node(
            'BatchNormalization', input_names, output_names,
            epsilon=func.eps,
            momentum=momentum,
            is_test=not chainer.config.train,
        ),
    elif opset_version == 7:
        return onnx_helper.make_node(
            'BatchNormalization', input_names, output_names,
            epsilon=func.eps,
            momentum=momentum,
        ),


@support((1, 6, 7))
def convert_FixedBatchNormalization(
        func, opset_version, input_names, output_names, context):
    return convert_BatchNormalization(
        func, opset_version, input_names, output_names, context)


@support((5, 10))
def convert_GroupNormalization(
        func, opset_version, input_names, output_names, context):
    # drop opset < 5, to reduce supporting cost of old Reshape op
    # calculation process is from
    # https://github.com/chainer/chainer/blob/v6.2.0/chainer/functions/normalization/group_normalization.py  # NOQA
    # support dynamic batch size, but channel size is expected to be fixed

    group = context.add_const(np.array(func.groups, dtype=np.int64), 'group')
    eps = context.add_const(np.array(func.eps, dtype=np.float32), 'eps')
    channel_size = context.add_const(
        np.array([func.inputs[0].shape[1]], dtype=np.int64), 'channel')

    neg_one = context.add_const(np.array([-1], dtype=np.int64), 'neg_one')

    gb = onnx_helper.GraphBuilder()

    # make reduced input
    original_shape = gb.op('Shape', [input_names[0]])
    batch_size = get_slice_node(
        gb, opset_version, context, [original_shape], [0], [0], [1], [1])
    batched_group = gb.op('Mul', [batch_size, group])
    reduce_shape = gb.op('Concat', [batched_group, neg_one], axis=0)
    reduced_x = gb.op('Reshape', [input_names[0], reduce_shape])

    # calculate mean, var and x_hat
    mean = gb.op('Unsqueeze', [
        gb.op('ReduceMean', [reduced_x], axes=[1], keepdims=0)], axes=[1])
    x_hat = gb.op('Sub', [reduced_x, mean])
    var = gb.op('Add', [
        gb.op(
            'ReduceMean', [gb.op('Mul', [x_hat, x_hat])], axes=[1],
            keepdims=0),
        eps])
    inv_std = gb.op('Unsqueeze', [
        gb.op('Reciprocal', [gb.op('Sqrt', [var])])], axes=[1])
    x_hat_ = gb.op('Mul', [x_hat, inv_std])

    # make out y
    groupless_shape = gb.op(
        'Concat', [batch_size, channel_size, neg_one], axis=0)
    y_org = gb.op('Reshape', [x_hat_, groupless_shape])

    # gamma/beta
    gamma = gb.op('Unsqueeze', [input_names[1]], axes=[1])
    beta = gb.op('Unsqueeze', [input_names[2]], axes=[1])
    y_g = gb.op('Mul', [y_org, gamma])
    y_b = gb.op('Add', [y_g, beta])
    gb.op('Reshape', [y_b, original_shape])

    return gb.nodes(output_names)


def convert_LocalResponseNormalization(
        func, opset_version, input_names, output_names, context):
    size = int(func.n)
    return onnx_helper.make_node(
        'LRN', input_names, output_names,
        alpha=float(func.alpha) * size,
        beta=float(func.beta),
        bias=float(func.k),
        size=size,
    ),


def convert_NormalizeL2(
        func, opset_version, input_names, output_names, context):
    if isinstance(func.axis, tuple) and len(func.axis) != 1:
        raise ValueError(
            'Normalization along with multiple axes ({}) are not supported in '
            'the ONNX\'s LpNormalization operator.'.format(func.axis))
    if abs(func.eps - 1e-5) > sys.float_info.epsilon:
        # default value of F.normaize eps is 1e-5
        raise ValueError(
            '\'eps\' is not supported in the ONNX\'s LpNormalization operator,'
            ' so that ONNX-Chainer does not accept custom values for \'eps\' '
            '({})'.format(func.eps))

    return onnx_helper.make_node(
        'LpNormalization', input_names, output_names,
        axis=int(func.axis[0]),
        p=2,
    ),
