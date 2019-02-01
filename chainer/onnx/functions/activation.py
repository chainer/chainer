from onnx import helper


def convert_ClippedReLU(func, onnx_op_name, opset_version, input_names,
                        output_names, parameters):
    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            min=0.0, max=func.cap,
            consumed_inputs=[1]
        ),
    elif opset_version == 6:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            min=0.0, max=func.cap,
        ),


def convert_ELU(func, onnx_op_name, opset_version, input_names, output_names,
                parameters):
    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            alpha=func.alpha,
        ),
    elif opset_version == 6:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            alpha=func.alpha
        ),


def convert_HardSigmoid(func, onnx_op_name, opset_version, input_names,
                        output_names, parameters):
    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            alpha=0.2,
            beta=0.5,
            consumed_inputs=[1],
        ),
    elif opset_version == 6:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            alpha=0.2,
            beta=0.5
        ),


def convert_LeakyReLU(func, onnx_op_name, opset_version, input_names,
                      output_names, parameters):
    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            alpha=func.slope,
            consumed_inputs=[1],
        ),
    elif opset_version == 6:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            alpha=func.slope
        ),


def convert_LogSoftmax(func, onnx_op_name, opset_version, input_names,
                       output_names, parameters):
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        axis=1
    ),


def convert_PReLUFunction(func, onnx_op_name, opset_version, input_names,
                          output_names, parameters):
    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names, consumed_inputs=[1]),
    elif opset_version == 6:
        return helper.make_node(onnx_op_name, input_names, output_names),
    elif opset_version == 7:
        return helper.make_node(onnx_op_name, input_names, output_names),


def convert_ReLU(func, onnx_op_name, opset_version, input_names, output_names,
                 parameters):
    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            consumed_inputs=[1]),
    elif opset_version == 6:
        return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Sigmoid(func, onnx_op_name, opset_version, input_names,
                    output_names, parameters):
    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            consumed_inputs=[1]),
    elif opset_version == 6:
        return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Softmax(func, onnx_op_name, opset_version, input_names,
                    output_names, parameters):
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        axis=func.axis
    ),


def convert_Softplus(func, onnx_op_name, opset_version, input_names,
                     output_names, parameters):
    return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Tanh(func, onnx_op_name, opset_version, input_names, output_names,
                 parameters):
    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            consumed_inputs=[1]),
    elif opset_version == 6:
        return helper.make_node(onnx_op_name, input_names, output_names),
