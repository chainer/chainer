import os

import onnx_chainer


TEST_OUT_DIR = 'out'


def gen_test_data_set(model, args, name, opset_version, **kwargs):
    model.xp.random.seed(42)
    test_path = os.path.join(
        TEST_OUT_DIR, 'opset{}'.format(opset_version), name)
    onnx_chainer.export_testcase(
        model, args, test_path, opset_version=opset_version, **kwargs)
    return test_path
