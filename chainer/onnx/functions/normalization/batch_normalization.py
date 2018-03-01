import os

import chainer
from onnx import helper
from onnx import numpy_helper

from onnx_chainer import mapping


def convert_BatchNormalization(
        func, input_names, param_names, parameters, input_tensors):

    layer_name = os.path.dirname(param_names[id(func.gamma)])

    # Add running_mean and running_var to graph
    param_names[id(func.running_mean)] = os.path.join(
        layer_name, 'running_mean')
    parameters.append(
        numpy_helper.from_array(
            func.running_mean,
            param_names[id(func.running_mean)]))
    input_tensors.append(
        helper.make_tensor_value_info(
            param_names[id(func.running_mean)],
            mapping.dtypes[func.running_mean.dtype],
            func.running_mean.shape)
    )

    param_names[id(func.running_var)] = os.path.join(
        layer_name, 'running_var')
    parameters.append(
        numpy_helper.from_array(
            func.running_var,
            param_names[id(func.running_var)]))
    input_tensors.append(
        helper.make_tensor_value_info(
            param_names[id(func.running_var)],
            mapping.dtypes[func.running_var.dtype],
            func.running_var.shape)
    )

    gamma_idx = input_names.index(id(func.gamma))
    input_names[gamma_idx] = param_names[id(func.gamma)]
    beta_idx = input_names.index(id(func.beta))
    input_names[beta_idx] = param_names[id(func.beta)]
    for i, input_name in enumerate(input_names):
        if type(input_name) is not str:
            input_names[i] = str(input_name)
    input_names.append(param_names[id(func.running_mean)])
    input_names.append(param_names[id(func.running_var)])

    layer_name = mapping.operators[func.__class__.__name__]
    unique_layer_name = os.path.dirname(input_names[1])
    out_names = [str(id(out())) for out in func.outputs]
    out_names += [
        os.path.join(unique_layer_name, 'mean'),
        os.path.join(unique_layer_name, 'var'),
        os.path.join(unique_layer_name, 'saved_mean'),
        os.path.join(unique_layer_name, 'saved_var')
    ]

    return helper.make_node(
        layer_name, input_names, out_names,
        epsilon=func.eps,
        is_test=not chainer.config.train,
        momentum=func.decay,
        spatial=True,
        consumed_inputs=[False, False, False, True, True],
    ),


def convert_FixedBatchNormalization(
        func, input_names, param_names, parameters, input_tensors):

    layer_name = os.path.dirname(param_names[id(func.gamma)])

    # Add avg_mean and avg_var to graph
    mean_id, var_id = input_names[3:]
    mean_arr, var_arr = [i.get_variable().array for i in func.inputs[3:]]

    param_names[mean_id] = os.path.join(layer_name, 'mean')
    parameters.append(
        numpy_helper.from_array(
            mean_arr,
            param_names[mean_id]
        )
    )
    input_tensors.append(
        helper.make_tensor_value_info(
            param_names[mean_id],
            mapping.dtypes[mean_arr.dtype],
            mean_arr.shape
        )
    )

    param_names[var_id] = os.path.join(layer_name, 'var')
    parameters.append(
        numpy_helper.from_array(
            var_arr,
            param_names[var_id]
        )
    )
    input_tensors.append(
        helper.make_tensor_value_info(
            param_names[var_id],
            mapping.dtypes[var_arr.dtype],
            var_arr.shape
        )
    )

    gamma_idx = input_names.index(id(func.gamma))
    input_names[gamma_idx] = param_names[id(func.gamma)]

    beta_idx = input_names.index(id(func.beta))
    input_names[beta_idx] = param_names[id(func.beta)]

    mean_idx = input_names.index(mean_id)
    input_names[mean_idx] = param_names[mean_id]

    var_idx = input_names.index(var_id)
    input_names[var_idx] = param_names[var_id]

    for i, input_name in enumerate(input_names):
        if type(input_name) is not str:
            input_names[i] = str(input_name)

    layer_name = mapping.operators[func.__class__.__name__]
    out_names = [str(id(out())) for out in func.outputs]

    return helper.make_node(
        layer_name, input_names, out_names,
        epsilon=func.eps,
        is_test=not chainer.config.train,
        momentum=0.9,
        spatial=True,
        consumed_inputs=[False, False, False, True, True],
    ),
