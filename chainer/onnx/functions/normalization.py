import sys

from onnx import helper

import chainer


def convert_BatchNormalization(func, onnx_op_name, opset_version, input_names,
                               output_names, parameters):
    # Add running_mean and running_var to graph
    running_mean = chainer.Parameter(func.running_mean)
    parameters.append(running_mean)
    input_names.append(str(id(running_mean)))

    running_var = chainer.Parameter(func.running_var)
    parameters.append(running_var)
    input_names.append(str(id(running_var)))

    unique_layer_name = '{}_{}'.format(func.__class__.__name__, str(id(func)))
    output_names += [
        unique_layer_name + '_mean',
        unique_layer_name + '_var',
        unique_layer_name + '_saved_mean',
        unique_layer_name + '_saved_var'
    ]

    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            epsilon=func.eps,
            momentum=func.decay,
            is_test=not chainer.config.train,
            consumed_inputs=[False, False, False, True, True],
        ),
    elif opset_version == 6:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            epsilon=func.eps,
            momentum=func.decay,
            is_test=not chainer.config.train,
        ),
    elif opset_version == 7:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            epsilon=func.eps,
            momentum=func.decay,
        ),


def convert_FixedBatchNormalization(func, onnx_op_name, opset_version,
                                    input_names, output_names, parameters):
    # Add avg_mean and avg_var to graph
    mean_arr, var_arr = [i.get_variable().array for i in func.inputs[3:]]

    mean_arr_param = chainer.Parameter(mean_arr)
    parameters.append(mean_arr_param)
    input_names[3] = str(id(mean_arr_param))

    var_arr_param = chainer.Parameter(var_arr)
    parameters.append(var_arr_param)
    input_names[4] = str(id(var_arr_param))

    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            epsilon=func.eps,
            momentum=0.,
            is_test=not chainer.config.train,
            consumed_inputs=[False, False, False, True, True],
        ),
    elif opset_version == 6:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            epsilon=func.eps,
            momentum=0.,
            is_test=not chainer.config.train,
        ),
    elif opset_version == 7:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            epsilon=func.eps,
            momentum=0.,
        ),


def convert_LocalResponseNormalization(func, onnx_op_name, opset_version,
                                       input_names, output_names, parameters):
    if opset_version == 1:
        size = int(func.n)
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            alpha=float(func.alpha) * size,
            beta=float(func.beta),
            bias=float(func.k),
            size=size,
        ),


def convert_NormalizeL2(func, onnx_op_name, opset_version, input_names,
                        output_names, parameters):
    if isinstance(func.axis, tuple) and len(func.axis) != 1:
        raise ValueError(
            'multiple axes ({}) is not supported in ONNX\'s LpNormalization '
            'operation'.format(func.axis))
    if abs(func.eps - 1e-5) > sys.float_info.epsilon:
        # default value of F.normaize eps is 1e-5
        raise ValueError(
            'eps is not supported in ONNX\'s LpNormalization, ONNX-Chainer '
            'does not follow customized eps ({})'.format(func.eps))
    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            axis=int(func.axis[0]),
            p=2,
        ),
