import glob
import os
import warnings

import numpy as np
import onnx

try:
    import onnxruntime as rt
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    warnings.warn(
        'ONNXRuntime is not installed. Please install it to use '
        ' the testing utility for ONNX-Chainer\'s converters.',
        ImportWarning)
    ONNXRUNTIME_AVAILABLE = False


def load_test_data(data_dir, input_names, output_names):
    inout_values = []
    for kind, names in [('input', input_names), ('output', output_names)]:
        names = list(names)
        values = {}
        for pb in sorted(
                glob.glob(os.path.join(data_dir, '{}_*.pb'.format(kind)))):
            tensor = onnx.load_tensor(pb)
            if tensor.name in names:
                name = tensor.name
                names.remove(name)
            else:
                name = names.pop(0)
            values[name] = onnx.numpy_helper.to_array(tensor)
        inout_values.append(values)
    return tuple(inout_values)


def check_model_expect(test_path, input_names=None, rtol=1e-5, atol=1e-5):
    if not ONNXRUNTIME_AVAILABLE:
        raise ImportError('ONNX Runtime is not found on checking module.')

    model_path = os.path.join(test_path, 'model.onnx')
    with open(model_path, 'rb') as f:
        onnx_model = onnx.load_model(f)
    sess = rt.InferenceSession(onnx_model.SerializeToString())
    rt_input_names = [value.name for value in sess.get_inputs()]
    rt_output_names = [value.name for value in sess.get_outputs()]

    # To detect unexpected inputs created by exporter, check input names
    if input_names is not None:
        assert list(sorted(input_names)) == list(sorted(rt_input_names))

    test_data_sets = sorted([
        p for p in os.listdir(test_path) if p.startswith('test_data_set_')])
    for test_data in test_data_sets:
        test_data_path = os.path.join(test_path, test_data)
        assert os.path.isdir(test_data_path)
        inputs, outputs = load_test_data(
            test_data_path, rt_input_names, rt_output_names)

        rt_out = sess.run(list(outputs.keys()), inputs)
        for cy, my in zip(outputs.values(), rt_out):
            np.testing.assert_allclose(cy, my, rtol=rtol, atol=atol)
