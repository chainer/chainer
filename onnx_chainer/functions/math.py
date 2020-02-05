import numpy as np
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from onnx_chainer.functions.opset_version import support
from onnx_chainer import onnx_helper


@support((1, 6, 7))
def convert_Add(func, opset_version, input_names, output_names, context):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Add', input_names, output_names, consumed_inputs=[1, 1]),
    elif opset_version == 6 or opset_version == 7:
        return onnx_helper.make_node('Add', input_names, output_names),


@support((1, 6, 7))
def convert_AddConstant(
        func, opset_version, input_names, output_names, context):
    value_name = context.add_const(
        np.array(func.value, dtype=func.inputs[0].dtype), 'value')
    input_names.append(value_name)

    if opset_version == 1:
        return onnx_helper.make_node(
            'Add', input_names, output_names, consumed_inputs=[1, 1]),
    elif opset_version == 6 or opset_version == 7:
        return onnx_helper.make_node('Add', input_names, output_names),


@support((1, 6, 7))
def convert_Sub(func, opset_version, input_names, output_names, context):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Sub', input_names, output_names, consumed_inputs=[1, 1]),
    elif opset_version == 6 or opset_version == 7:
        return onnx_helper.make_node('Sub', input_names, output_names),


@support((1, 6, 7))
def convert_SubFromConstant(
        func, opset_version, input_names, output_names, context):
    value_name = context.add_const(
        np.array(func.value, dtype=func.inputs[0].dtype), 'value')
    input_names[:0] = [value_name]

    if opset_version == 1:
        return onnx_helper.make_node(
            'Sub', input_names, output_names, consumed_inputs=[1, 1]),
    elif opset_version == 6 or opset_version == 7:
        return onnx_helper.make_node('Sub', input_names, output_names),


@support((1, 6, 7))
def convert_Mul(func, opset_version, input_names, output_names, context):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Mul', input_names, output_names, consumed_inputs=[1, 1]),
    elif opset_version == 6 or opset_version == 7:
        return onnx_helper.make_node('Mul', input_names, output_names),


@support((1, 6, 7))
def convert_MulConstant(
        func, opset_version, input_names, output_names, context):
    value_name = context.add_const(
        np.array(func.value, dtype=func.inputs[0].dtype), 'value')
    input_names.append(value_name)

    if opset_version == 1:
        return onnx_helper.make_node(
            'Mul', input_names, output_names, consumed_inputs=[1, 1]),
    elif opset_version == 6 or opset_version == 7:
        return onnx_helper.make_node('Mul', input_names, output_names),


@support((1, 6))
def convert_Neg(func, opset_version, input_names, output_names, context):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Neg', input_names, output_names, consumed_inputs=[1, 1]),
    elif opset_version == 6:
        return onnx_helper.make_node('Neg', input_names, output_names),


@support((1, 6, 7))
def convert_Div(func, opset_version, input_names, output_names, context):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Div', input_names, output_names, consumed_inputs=[1, 1]),
    elif opset_version == 6 or opset_version == 7:
        return onnx_helper.make_node('Div', input_names, output_names),


@support((1, 6, 7))
def convert_DivFromConstant(
        func, opset_version, input_names, output_names, context):
    value_name = context.add_const(
        np.array(func.value, dtype=func.inputs[0].dtype), 'value')
    input_names[:0] = [value_name]

    if opset_version == 1:
        return onnx_helper.make_node(
            'Div', input_names, output_names, consumed_inputs=[1, 1]),
    elif opset_version == 6 or opset_version == 7:
        return onnx_helper.make_node('Div', input_names, output_names),


@support((1, 6))
def convert_Absolute(func, opset_version, input_names, output_names, context):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Abs', input_names, output_names, consumed_inputs=[1]),
    elif opset_version == 6:
        return onnx_helper.make_node('Abs', input_names, output_names),


@support((7,))
def convert_Arccos(func, opset_version, input_names, output_names, context):
    return onnx_helper.make_node('Acos', input_names, output_names),


@support((7,))
def convert_Arcsin(func, opset_version, input_names, output_names, context):
    return onnx_helper.make_node('Asin', input_names, output_names),


@support((7,))
def convert_Arctan(func, opset_version, input_names, output_names, context):
    return onnx_helper.make_node('Atan', input_names, output_names),


@support((1, 7))
def convert_PowConstVar(
        func, opset_version, input_names, output_names, context):
    value_name = context.add_const(
        np.array(func.value, dtype=func.inputs[0].dtype), 'value')
    input_names.insert(0, value_name)

    if opset_version == 1 or opset_version == 7:
        return onnx_helper.make_node('Pow', input_names, output_names),


@support((1, 7))
def convert_PowVarConst(
        func, opset_version, input_names, output_names, context):
    value_name = context.add_const(
        np.array(func.value, dtype=func.inputs[0].dtype), 'value')
    input_names.append(value_name)

    if opset_version == 1 or opset_version == 7:
        return onnx_helper.make_node('Pow', input_names, output_names),


@support((1, 7))
def convert_PowVarVar(
        func, opset_version, input_names, output_names, context):
    if opset_version == 1 or opset_version == 7:
        return onnx_helper.make_node('Pow', input_names, output_names),


@support((1, 6, 11))
def convert_Clip(func, opset_version, input_names, output_names, context):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Clip', input_names, output_names,
            max=func.x_max,
            min=func.x_min,
            consumed_inputs=[1]
        ),
    elif opset_version == 6:
        return onnx_helper.make_node(
            'Clip', input_names, output_names,
            max=func.x_max,
            min=func.x_min,
        ),
    elif opset_version == 11:
        min_name = context.add_const(
            np.array(func.x_min, dtype=np.float32), 'x_min')
        max_name = context.add_const(
            np.array(func.x_max, dtype=np.float32), 'x_max')
        input_names.extend([min_name, max_name])
        return onnx_helper.make_node('Clip', input_names, output_names),


@support((7,))
def convert_Cos(func, opset_version, input_names, output_names, context):
    return onnx_helper.make_node('Cos', input_names, output_names),


@support((9,))
def convert_Cosh(func, opset_version, input_names, output_names, context):
    return onnx_helper.make_node('Cosh', input_names, output_names),


@support((1, 6))
def convert_Exp(func, opset_version, input_names, output_names, context):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Exp', input_names, output_names, consumed_inputs=[1, 1]),
    elif opset_version == 6:
        return onnx_helper.make_node('Exp', input_names, output_names),


def convert_Identity(func, opset_version, input_names, output_names, context):
    return onnx_helper.make_node('Identity', input_names, output_names),


def convert_MatMul(func, opset_version, input_names, output_names, context):
    ndim_a = len(func.inputs[0].shape)
    ndim_b = len(func.inputs[1].shape)

    gb = onnx_helper.GraphBuilder()
    if ndim_a > 1 and func.transa:
        perm = list(range(ndim_a))
        perm[-1], perm[-2] = perm[-2], perm[-1]
        input_names[0] = gb.op('Transpose', [input_names[0]], perm=perm)
    if ndim_b > 1 and func.transb:
        perm = list(range(ndim_b))
        perm[-1], perm[-2] = perm[-2], perm[-1]
        input_names[1] = gb.op('Transpose', [input_names[1]], perm=perm)
    gb.op('MatMul', input_names)
    return gb.nodes(output_names)


@support((1, 6, 8))
def convert_Maximum(func, opset_version, input_names, output_names, context):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Max', input_names, output_names, consumed_inputs=[1, 1]),
    elif opset_version == 6 or opset_version == 8:
        return onnx_helper.make_node('Max', input_names, output_names),


@support((1, 6, 8))
def convert_Minimum(func, opset_version, input_names, output_names, context):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Min', input_names, output_names, consumed_inputs=[1, 1]),
    elif opset_version == 6 or opset_version == 8:
        return onnx_helper.make_node('Min', input_names, output_names),


@support((7,))
def convert_Sin(func, opset_version, input_names, output_names, context):
    return onnx_helper.make_node('Sin', input_names, output_names),


@support((9,))
def convert_Sinh(func, opset_version, input_names, output_names, context):
    return onnx_helper.make_node('Sinh', input_names, output_names),


@support((1, 6))
def convert_Sqrt(func, opset_version, input_names, output_names, context):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Sqrt', input_names, output_names, consumed_inputs=[1, 1]),
    elif opset_version == 6:
        return onnx_helper.make_node('Sqrt', input_names, output_names),


def convert_RsqrtGPU(func, opset_version, input_names, output_names, context):
    gb = onnx_helper.GraphBuilder()
    sqrt_out = gb.op('Sqrt', input_names)
    gb.op('Reciprocal', [sqrt_out])
    return gb.nodes(output_names)


@support((6,))
def convert_Log(func, opset_version, input_names, output_names, context):
    return onnx_helper.make_node('Log', input_names, output_names),


def convert_LogSumExp(func, opset_version, input_names, output_names, context):
    # Use keepdims=False by default
    # since the chainer does not support keepdims option
    kwargs = {'keepdims': False}
    if hasattr(func, 'keepdims'):
        kwargs['keepdims'] = func.keepdims
    if func.axis is not None:
        kwargs['axes'] = func.axis
    return onnx_helper.make_node(
        'ReduceLogSumExp', input_names, output_names, **kwargs),


def convert_Max(func, opset_version, input_names, output_names, context):
    kwargs = {'keepdims': func.keepdims}
    if func.axis is not None:
        kwargs['axes'] = func.axis
    return onnx_helper.make_node(
        'ReduceMax', input_names, output_names, **kwargs),


def convert_Mean(func, opset_version, input_names, output_names, context):
    kwargs = {'keepdims': func.keepdims}
    if func.axis is not None:
        kwargs['axes'] = func.axis
    return onnx_helper.make_node(
        'ReduceMean', input_names, output_names, **kwargs),


def convert_Min(func, opset_version, input_names, output_names, context):
    kwargs = {'keepdims': func.keepdims}
    if func.axis is not None:
        kwargs['axes'] = func.axis
    return onnx_helper.make_node(
        'ReduceMin', input_names, output_names, **kwargs),


def convert_Prod(func, opset_version, input_names, output_names, context):
    kwargs = {'keepdims': func.keepdims}
    if func.axis is not None:
        kwargs['axes'] = func.axis
    return onnx_helper.make_node(
        'ReduceProd', input_names, output_names, **kwargs),


def convert_Sum(func, opset_version, input_names, output_names, context):
    kwargs = {'keepdims': func.keepdims}
    if func.axis is not None:
        kwargs['axes'] = func.axis
    return onnx_helper.make_node(
        'ReduceSum', input_names, output_names, **kwargs),


@support((7,))
def convert_Tan(func, opset_version, input_names, output_names, context):
    return onnx_helper.make_node('Tan', input_names, output_names),


@support((1, 6, 7))
def convert_LinearInterpolate(
        func, opset_version, input_names, output_names, context):
    typ = func.inputs[0].dtype if isinstance(
        func.inputs[0].dtype, np.dtype) else np.dtype(func.inputs[0].dtype)

    one_name = context.add_const(np.array(1, dtype=typ), 'one')

    kwargs = {'consumed_inputs': [1, 1]} if opset_version == 1 else {}
    kwargs2 = {} if opset_version >= 7 else {'broadcast': 1}

    gb = onnx_helper.GraphBuilder()
    p, x, y = input_names
    n1 = gb.op('Sub', [one_name, p], **kwargs, **kwargs2)
    n2 = gb.op('Mul', [p, x], **kwargs)
    n3 = gb.op('Mul', [n1, y], **kwargs)
    gb.op_output_named('Add', [n2, n3], output_names, **kwargs)

    return gb.nodes()


@support((1, 6, 7))
def convert_Square(func, opset_version, input_names, output_names, context):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Mul', [input_names[0], input_names[0]], output_names,
            consumed_inputs=[1, 1]),
    elif opset_version == 6 or opset_version == 7:
        return onnx_helper.make_node(
            'Mul', [input_names[0], input_names[0]], output_names),


@support((8,))
def convert_BroadcastTo(
        func, opset_version, input_names, output_names, context):
    shape_name = context.add_const(
        np.array(func._shape, dtype=np.int64), 'shape')
    input_names.append(shape_name)
    return onnx_helper.make_node('Expand', input_names, output_names),


def _argminmax_nodes(op_name, func, input_names, output_names, context):
    gb = onnx_helper.GraphBuilder()
    target_input_names = input_names
    axis = func.axis
    if axis is None:
        shape_name = context.add_const(np.array([-1], dtype=np.int64), 'shape')
        input_names.append(shape_name)
        target_input_names = [gb.op('Reshape', input_names)]
        axis = 0
    out = gb.op(op_name, target_input_names, axis=axis, keepdims=0)
    # Chainer's ArgMax always return value as int32
    # Cast spec is changed from opset6, this logic does not support ~opset5
    gb.op('Cast', [out], to=NP_TYPE_TO_TENSOR_TYPE[np.dtype('int32')])
    return gb.nodes(output_names)


@support((6,))
def convert_ArgMax(func, opset_version, input_names, output_names, context):
    return _argminmax_nodes('ArgMax', func, input_names, output_names, context)


@support((6,))
def convert_ArgMin(func, opset_version, input_names, output_names, context):
    return _argminmax_nodes('ArgMin', func, input_names, output_names, context)


@support((9,))
def convert_sign(func, opset_verseion, input_names, output_names, context):
    return onnx_helper.make_node('Sign', input_names, output_names),
