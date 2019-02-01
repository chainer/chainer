from onnx import helper

import chainer


def convert_Dropout(
        func, onnx_op_name, opset_version, input_names,
        output_names, parameters):
    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            is_test=0 if chainer.config.train else 1,
            ratio=func.dropout_ratio,
            consumed_inputs=[1]
        ),
    elif opset_version == 6:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            is_test=0 if chainer.config.train else 1,
            ratio=func.dropout_ratio,
        ),
    elif opset_version == 7:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            ratio=func.dropout_ratio,
        ),
