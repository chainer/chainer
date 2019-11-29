import warnings

import chainer
import numpy as np
import onnx
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from onnx_chainer.functions.opset_version import support
from onnx_chainer import onnx_helper


TENSOR_TYPE_TO_NAME = {
    0: 'UNDEFINED',
    1: 'FLOAT',
    2: 'UINT8',
    3: 'INT8',
    4: 'UINT16',
    5: 'INT16',
    6: 'INT32',
    7: 'INT64',
    8: 'STRING',
    9: 'BOOL',
    10: 'FLOAT16',
    11: 'DOUBLE',
    12: 'UINT32',
    13: 'UINT64',
    14: 'COMPLEX64',
    15: 'COMPLEX128',
}


@support((1, 6))
def convert_Cast(func, opset_version, input_names, output_names, context):
    typ = func.type if isinstance(func.type, np.dtype) else np.dtype(func.type)
    if opset_version == 1:
        return onnx_helper.make_node(
            'Cast', input_names, output_names,
            to=TENSOR_TYPE_TO_NAME[NP_TYPE_TO_TENSOR_TYPE[typ]]
        ),
    elif opset_version == 6:
        return onnx_helper.make_node(
            'Cast', input_names, output_names,
            to=NP_TYPE_TO_TENSOR_TYPE[typ]
        ),


@support((1, 4))
def convert_Concat(func, opset_version, input_names, output_names, context):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Concat', input_names, output_names,
            axis=func.axis
        ),
    elif opset_version == 4:
        return onnx_helper.make_node(
            'Concat', input_names, output_names,
            axis=func.axis
        ),


def convert_Copy(func, opset_version, input_names, output_names, context):
    return onnx_helper.make_node(
        'Identity', input_names, output_names
    ),


def convert_Depth2Space(
        func, opset_version, input_names, output_names, context):
    return onnx_helper.make_node(
        'DepthToSpace', input_names, output_names,
        blocksize=func.r
    ),


def get_slice_node(
        gb, opset_version, context, input_names, axes, starts, ends, steps):
    if opset_version < 11 and any([i != 1 for i in steps]):
        raise ValueError(
            'GetItem with n-step slicing is supported from opset11, '
            'opset{} is not supported'.format(opset_version))

    if opset_version < 10:
        return gb.op(
            'Slice', input_names, axes=axes, starts=starts, ends=ends)
    else:
        inputs = [('starts', starts), ('ends', ends), ('axes', axes)]
        if opset_version > 10:
            inputs.append(('steps', steps))
        for name, values in inputs:
            param_name = context.add_const(
                np.asarray(list(values), dtype=np.int64), name)
            input_names.append(param_name)
        return gb.op('Slice', input_names)


def _to_ndarray(x, dtype=np.int64):
    if isinstance(x, list):
        return np.array(x, dtype=dtype)
    else:
        return chainer.cuda.to_cpu(x).astype(dtype)


@support((1, 10, 11))
def convert_GetItem(func, opset_version, input_names, output_names, context):
    x = func.inputs[0]
    axes, starts, ends, steps = [], [], [], []
    squeeze_idxs, unsqueeze_idxs = [], []
    skipped = 0  # when set ellipsis, need to skip index rolling

    prev_gathered_axis = -1
    gather_axis = -1
    gather_idx = None  # when GatherND, set first array for broadcasting
    gather_nd_idx = None
    is_used_slice_whole = False  # GatherND does not support axis, need to care

    for i, idx in enumerate(func.slices):
        # axis means the index of input x, adjust None and Ellipsis counts
        axis = i - len(unsqueeze_idxs) + skipped
        if isinstance(idx, slice):
            if idx.start is None and idx.stop is None and idx.step is None:
                is_used_slice_whole = True
                continue
            axes.append(axis)
            step = 1 if idx.step is None else idx.step
            steps.append(step)
            if step < 0:
                starts.append(
                    np.iinfo(np.int64).max if idx.start is None else idx.start)
                ends.append(
                    np.iinfo(np.int64).min if idx.stop is None else idx.stop)
            else:
                starts.append(0 if idx.start is None else idx.start)
                ends.append(
                    np.iinfo(np.int64).max if idx.stop is None else idx.stop)
        elif isinstance(idx, int):
            axes.append(axis)
            steps.append(1)
            if idx == -1:
                starts.append(idx)
                ends.append(np.iinfo(np.int64).max)
            else:
                starts.append(idx)
                ends.append(idx+1)
            squeeze_idxs.append(axis)
        elif isinstance(idx, np.ndarray) and idx.ndim == 0:
            scalar_idx = idx.item()
            axes.append(axis)
            starts.append(scalar_idx)
            ends.append(scalar_idx+1)
            steps.append(1)
            squeeze_idxs.append(axis)
        elif idx is None:
            unsqueeze_idxs.append(i - len(squeeze_idxs) + skipped)
        elif idx is Ellipsis:
            # calculate rest slice number except None, GetItem does not allow
            # multiple Ellipsis, so ignore latter Ellipsis count
            rest_slice_len = len(
                [idx_ for idx_ in func.slices[i+1:] if idx_ is not None])
            assert skipped == 0
            skipped = len(x.shape) - axis - rest_slice_len - 1
        elif isinstance(idx, (list,) + chainer.get_array_types()):
            if prev_gathered_axis >= 0:
                if (i - 1) != prev_gathered_axis:
                    raise ValueError(
                        'ONNX-Chainer does not support non-consecutive'
                        'multiple advanced indexing')
                if is_used_slice_whole:
                    raise ValueError(
                        'ONNX-Chainer does not support whole indexing(`[:]`)'
                        'in front of multiple advanced indexing')
                if unsqueeze_idxs:
                    raise ValueError(
                        'ONNX-Chainer does not support new axis in front of '
                        'multiple advanced indexing')
                # multiple advanced index, convert to GatherND
                idx_array = _to_ndarray(idx)
                base_idx = gather_idx if gather_nd_idx is None else\
                    gather_nd_idx
                gather_nd_idx = np.vstack((base_idx, idx_array))
                prev_gathered_axis = i
            else:
                # convert to Gather, if next index is also list, change to
                # GatherND
                gather_axis = axis - len(squeeze_idxs) + len(unsqueeze_idxs)
                gather_idx = _to_ndarray(idx)
                prev_gathered_axis = i
        else:
            raise ValueError(
                'GetItem with type {} cannot handle in ONNX Slice, so that '
                'ONNX-Chainer does not accept the type'.format(type(idx)))

    gb = onnx_helper.GraphBuilder()
    slice_output = input_names
    if axes:
        output = get_slice_node(
            gb, opset_version, context, slice_output, axes, starts, ends,
            steps)
        slice_output = [output]
    if squeeze_idxs:
        output = gb.op('Squeeze', slice_output, axes=squeeze_idxs)
        slice_output = [output]
    if unsqueeze_idxs:
        output = gb.op('Unsqueeze', slice_output, axes=unsqueeze_idxs)
        slice_output = [output]

    if gather_nd_idx is not None:
        if opset_version < 11:
            raise ValueError(
                'ONNX-Chainer supports multiple advanced indexing from opset11'
                ', opset{} is not supported'.format(opset_version))
        gather_nd_idx_name = context.add_const(gather_nd_idx.T, 'indices')
        slice_output.append(gather_nd_idx_name)
        gb.op('GatherND', slice_output)
    elif gather_idx is not None:
        gather_idx_name = context.add_const(gather_idx, 'indices')
        slice_output.append(gather_idx_name)
        gb.op('Gather', slice_output, axis=gather_axis)

    return gb.nodes(output_names=output_names)


@support((9, 11))
def convert_SelectItem(func, opset_version, input_names, output_names,
                       context):
    gb = onnx_helper.GraphBuilder()

    if opset_version >= 11:
        t = gb.op('Unsqueeze', [input_names[1]], axes=[1])
        out = gb.op('GatherElements', [input_names[0], t], axis=1)
        gb.op('Squeeze', [out], axes=[1])
    else:
        data, target_idxs = input_names
        target_idxs = gb.op('Cast', [target_idxs],
                            to=NP_TYPE_TO_TENSOR_TYPE[np.dtype('int64')])
        n_rows = gb.op('Shape', [target_idxs])

        # This is an equivalent of using Range.
        one_1 = onnx.helper.make_tensor(
            'one_1', onnx.TensorProto.FLOAT, [1], [1])
        ones = gb.op('ConstantOfShape', [n_rows], value=one_1)
        row_idxs = gb.op('Squeeze', [gb.op('NonZero', [ones])])

        data_shape = gb.op('Shape', [data])
        one_2 = context.add_const(np.array([1]), 'one_2')
        n_cols = gb.op('Gather', [data_shape, one_2], axis=0)

        data = gb.op('Squeeze', [gb.op('Flatten', [data], axis=2)])
        target_idxs = gb.op(
            'Add', [target_idxs, gb.op('Mul', [row_idxs, n_cols])])
        gb.op('Gather', [data, target_idxs], axis=0)

    return gb.nodes(output_names)


@support((1, 2, 11))
def convert_Pad(func, opset_version, input_names, output_names, context):
    if func.mode not in ['constant', 'reflect', 'edge']:
        raise ValueError(
            '{} mode is not supported in ONNX\'s Pad operation'.format(
                func.mode))

    pad_begin = []
    pad_end = []
    pad_bw = func.pad_bw
    if pad_bw.ndim == 1:
        pad_bw = np.tile(pad_bw, (len(func.inputs[0].shape), 1))
    for pp in pad_bw.tolist():
        pad_begin.append(pp[0])
        pad_end.append(pp[1])
    pad = pad_begin + pad_end

    constant_value = func.keywords.get('constant_values', None)
    if constant_value is not None:
        # 'constant_values' only accepts int or array-like on Chainer
        if not isinstance(constant_value, int) and len(constant_value) > 1:
            raise ValueError(
                'ONNX doesn\'t support multiple constant values for Pad '
                'operation')
        elif not isinstance(constant_value, int):
            constant_value = float(constant_value[0])
        else:
            constant_value = float(constant_value)

    if opset_version == 1:
        kwargs = {
            'mode': func.mode,
            'paddings': pad,
        }
        if constant_value is not None:
            kwargs['value'] = constant_value
    elif opset_version == 2:
        kwargs = {
            'mode': func.mode,
            'pads': pad,
        }
        if constant_value is not None:
            kwargs['value'] = constant_value
    elif opset_version == 11:
        pads_name = context.add_const(np.array(pad, dtype=np.int64), 'pads')
        input_names.append(pads_name)
        if constant_value is not None:
            constant_value_name = context.add_const(
                np.array(constant_value, dtype=np.float32), 'constant_value')
            input_names.append(constant_value_name)
        kwargs = {'mode': func.mode}

    return onnx_helper.make_node('Pad', input_names, output_names, **kwargs),


@support((9, 11))
def convert_Permutate(func, opset_version, input_names, output_names, context):
    gb = onnx_helper.GraphBuilder()
    indices_name = context.get_name(func.indices)
    if func.inv:
        empty = context.add_const(
            np.zeros(dtype=np.int64, shape=func.indices.shape), 'empty')
        r = context.add_const(np.arange(len(func.indices), dtype=np.int64),
                              'range')
        op = 'ScatterElements' if opset_version == 11 else 'Scatter'
        indices_name = gb.op(op, [empty, indices_name, r])
    input_names.append(indices_name)
    gb.op_output_named('Gather', input_names, output_names, axis=func.axis)
    return gb.nodes()


@support((1, 5))
def convert_Reshape(func, opset_version, input_names, output_names, context):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Reshape', input_names, output_names,
            shape=func.shape
        ),
    elif opset_version == 5:
        if hasattr(func, 'shape'):
            # if the function has shape parameter, means not dynamic
            assert len(input_names) == 1
            shape_name = context.add_const(
                np.asarray(list(func.shape), dtype=np.int64), 'shape')
            input_names.append(shape_name)
        else:
            if len(input_names) != 2:
                raise ValueError('shape must be set as parameter or 2nd input')

        return onnx_helper.make_node(
            'Reshape', input_names, output_names,
        ),


def convert_Space2Depth(
        func, opset_version, input_names, output_names, context):
    return onnx_helper.make_node(
        'SpaceToDepth', input_names, output_names,
        blocksize=func.r
    ),


@support((1, 2))
def convert_SplitAxis(func, opset_version, input_names, output_names, context):
    if func.indices is not None:
        indices_or_sections = func.indices
    else:
        indices_or_sections = func.sections

    total = func.inputs[0].shape[func.axis]
    if hasattr(indices_or_sections, '__iter__'):
        split = []
        prev_i = 0
        for i in indices_or_sections:
            split.append(i - prev_i)
            prev_i = i
        split.append(total - prev_i)
    else:
        length = total // indices_or_sections
        split = [length for _ in range(indices_or_sections)]

    assert len(output_names) == len(split)
    if opset_version == 1:
        return onnx_helper.make_node(
            'Split', input_names, output_names,
            axis=func.axis,
            split=split
        ),
    elif opset_version == 2:
        return onnx_helper.make_node(
            'Split', input_names, output_names,
            axis=func.axis,
            split=split
        ),


def convert_Squeeze(func, opset_version, input_names, output_names, context):
    if func.axis is None:
        axis = []
        for i, s in enumerate(func.inputs[0].shape):
            if s == 1:
                axis.append(i)
    else:
        axis = func.axis

    return onnx_helper.make_node(
        'Squeeze', input_names, output_names,
        axes=axis
    ),


def convert_Swapaxes(func, opset_version, input_names, output_names, context):
    perm = list(range(len(func.inputs[0].shape)))
    perm[func.axis1], perm[func.axis2] = perm[func.axis2], perm[func.axis1]

    return onnx_helper.make_node(
        'Transpose', input_names, output_names, perm=perm
    ),


@support((1, 6))
def convert_Tile(func, opset_version, input_names, output_names, context):
    # Add tiles and axis to graph
    if isinstance(func.reps, int):
        func.reps = [func.reps]
    tiles_name = context.add_const(
        np.asarray(func.reps, dtype=np.int64), 'tiles')
    input_names.append(tiles_name)

    # In operater version = 1, axis also should be given
    if opset_version == 1:
        axis_name = context.add_const(
            np.array([i for i, _ in enumerate(func.reps)], dtype=np.float32),
            'axis')
        input_names.append(axis_name)

    return onnx_helper.make_node('Tile', input_names, output_names),


def convert_Transpose(func, opset_version, input_names, output_names, context):

    if func.axes is None:
        node = onnx_helper.make_node('Transpose', input_names, output_names)
    else:
        node = onnx_helper.make_node(
            'Transpose', input_names, output_names,
            perm=func.axes
        )

    return node,


def convert_ExpandDims(
        func, opset_version, input_names, output_names, context):
    axis = func.axis
    if axis < 0:
        axis = len(func.inputs[0].shape) + 1 + axis

    return onnx_helper.make_node(
        'Unsqueeze', input_names, output_names, axes=[axis]),


@support((9,))
def convert_Where(func, opset_version, input_names, output_names, context):
    input_names.insert(0, context.get_name(func.condition))
    return onnx_helper.make_node('Where', input_names, output_names),


@support((7, 9, 10, 11))
def convert_Repeat(func, opset_version, input_names, output_names, context):
    repeats = func.repeats
    if len(repeats) > 1:
        raise NotImplementedError(
            'ONNX-Chainer currently does not support elementwise repeat')

    gb = onnx_helper.GraphBuilder()
    inputs = list(input_names)
    axis = func.axis
    if axis is None:
        shape_name = context.add_const(np.array([-1], dtype=np.int64), 'shape')
        input_names.append(shape_name)
        inputs = [gb.op('Reshape', input_names)]
        scales = [float(repeats[0])]
    else:
        scales = [1.0] * func.inputs[0].data.ndim
        scales[axis] = float(repeats[0])

    if opset_version == 7:
        gb.op_output_named('Upsample', inputs, output_names, scales=scales)
        return gb.nodes()

    scales_name = context.add_const(
        np.array(scales, dtype=np.float32), 'scales')

    if opset_version in [9, 10]:
        inputs.append(scales_name)
        op = 'Upsample' if opset_version == 9 else 'Resize'
        gb.op_output_named(op, inputs, output_names)
        return gb.nodes()

    if opset_version == 11:
        roi = context.add_const(np.array([]), 'roi')
        inputs.extend([roi, scales_name])
        gb.op_output_named('Resize', inputs, output_names)
        return gb.nodes()


@support((7, 9, 10, 11))
def convert_ResizeImages(
        func, opset_version, input_names, output_names, context):

    warnings.warn(
        '`resize_images` is mapped to `Upsampling` ONNX op with bilinear '
        'interpolation. '
        'Behavior of bilinear interpolation differs from each implementation. '
        'See the issue https://github.com/chainer/onnx-chainer/issues/147 '
        'for details.',
        UserWarning)

    outsize = (func.out_H, func.out_W)

    h, w = func.inputs[0].shape[2:]

    # Compute scaling factor.
    # NOTE(syoyo): Despite of its name, `Upsample` onnx op will downsample
    # images when scale value is less than 1.0
    scales = [1.0, 1.0, float(outsize[0]) / float(h),
              float(outsize[1]) / float(w)]

    if (scales[2] < 1.0e-8) and (scales[3] < 1.0e-8):
        raise ValueError(
            'scaling factor is too small or zero. scales for h = {}, scales '
            'for w = {}'.format(scales[2], scales[3]))

    # resize_images in Chainer only supports bilinear interpolation
    # Actually this will be mapped to 'bilinear' in onnxruntime
    mode = 'linear'
    if opset_version == 7:
        return onnx_helper.make_node('Upsample', input_names, output_names,
                                     scales=scales, mode=mode),

    scales_name = context.add_const(
        np.array(scales, dtype=np.float32), 'scales')
    if opset_version in [9, 10]:
        input_names.append(scales_name)
        op = 'Upsample' if opset_version == 9 else 'Resize'
        return onnx_helper.make_node(op, input_names, output_names,
                                     mode=mode),

    if opset_version == 11:
        roi_name = context.add_const(np.array([]), 'roi')
        input_names.extend([roi_name, scales_name])
        return onnx_helper.make_node(
            'Resize', input_names, output_names, mode=mode),


def convert_Stack(func, opset_version, input_names, output_names, context):
    gb = onnx_helper.GraphBuilder()
    axis = func.axis
    if axis < 0:
        axis = len(func.inputs[0].shape) + 1 + axis

    # To use concat op, reshape every inputs add new axes
    inputs = [gb.op('Unsqueeze', [name], axes=[axis]) for name in input_names]
    gb.op_output_named('Concat', inputs, output_names, axis=axis)
    return gb.nodes()


def convert_Hstack(func, opset_version, input_names, output_names, context):
    gb = onnx_helper.GraphBuilder()
    input0_ndim = len(func.inputs[0].shape)
    inputs = input_names
    axis = 1
    if input0_ndim == 0:
        inputs = [gb.op('Unsqueeze', [name], axes=[0]) for name in input_names]
        axis = 0
    elif input0_ndim == 1:
        axis = 0
    gb.op_output_named('Concat', inputs, output_names, axis=axis)
    return gb.nodes()


def convert_Vstack(func, opset_version, input_names, output_names, context):
    gb = onnx_helper.GraphBuilder()
    input0_ndim = len(func.inputs[0].shape)
    inputs = input_names
    if input0_ndim == 0:
        inputs = [gb.op('Unsqueeze', [name], axes=[0, 1]) for
                  name in input_names]
    elif input0_ndim == 1:
        inputs = [gb.op('Unsqueeze', [name], axes=[0]) for name in input_names]
    gb.op_output_named('Concat', inputs, output_names, axis=0)
    return gb.nodes()


def convert_Dstack(func, opset_version, input_names, output_names, context):
    gb = onnx_helper.GraphBuilder()
    input0_ndim = len(func.inputs[0].shape)
    inputs = input_names
    if input0_ndim == 0:
        inputs = [gb.op('Unsqueeze', [name], axes=[0, 1, 2]) for
                  name in input_names]
    elif input0_ndim == 1:
        inputs = [gb.op('Unsqueeze', [name], axes=[0, 2]) for
                  name in input_names]
    elif input0_ndim == 2:
        inputs = [gb.op('Unsqueeze', [name], axes=[2]) for name in input_names]
    gb.op_output_named('Concat', inputs, output_names, axis=2)
    return gb.nodes()


def convert_Separate(func, opset_version, input_names, output_names, context):
    gb = onnx_helper.GraphBuilder()
    split_outs = gb.op(
        'Split', input_names, num_outputs=len(output_names), axis=func.axis)
    if len(output_names) == 1:
        split_outs = [split_outs]
    for i, node_name in enumerate(split_outs):
        gb.op_output_named(
            'Squeeze', [node_name], [output_names[i]], axes=[func.axis])
    return gb.nodes()


def convert_Shape(func, opset_version, input_names, output_names, context):
    return onnx_helper.make_node('Shape', input_names, output_names),


def convert_Moveaxis(func, opset_version, input_names, output_names, context):

    ndim = len(func.inputs[0].shape)
    source = [a % ndim for a in func.source]
    destination = [a % ndim for a in func.destination]

    order = [n for n in range(ndim) if n not in source]
    for dest, src in sorted(zip(destination, source)):
        order.insert(dest, src)

    node = onnx_helper.make_node('Transpose', input_names, output_names,
                                 perm=order)

    return node,


def convert_Rollaxis(func, opset_version, input_names, output_names, context):
    ndim = len(func.inputs[0].shape)

    order = list(range(ndim))
    order.remove(func.axis)
    order.insert(func.start, func.axis)

    node = onnx_helper.make_node('Transpose', input_names, output_names,
                                 perm=order)

    return node,


def convert_TransposeSequence(
        func, opset_version, input_names, output_names, context):
    if len(input_names) == 1:
        return onnx_helper.make_node(
            'Split', input_names, output_names, axis=0),
    elif len(output_names) == 1:
        return onnx_helper.make_node(
            'Concat', input_names, output_names, axis=0),
    else:
        raise ValueError(
            'ONNX-Chainer can convert TransposeSequence only when input '
            'or output length is 1')
