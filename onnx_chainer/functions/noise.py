import chainer

from onnx_chainer.functions.opset_version import support
from onnx_chainer import onnx_helper


@support((1, 6, 7))
def convert_Dropout(func, opset_version, input_names, output_names, context):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Dropout', input_names, output_names,
            is_test=0 if chainer.config.train else 1,
            ratio=func.dropout_ratio,
            consumed_inputs=[1]
        ),
    elif opset_version == 6:
        return onnx_helper.make_node(
            'Dropout', input_names, output_names,
            is_test=0 if chainer.config.train else 1,
            ratio=func.dropout_ratio,
        ),
    elif opset_version == 7:
        return onnx_helper.make_node(
            'Dropout', input_names, output_names,
            ratio=func.dropout_ratio,
        ),
