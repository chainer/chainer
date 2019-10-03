import chainer
import onnx
import pytest

import onnx_chainer


def pytest_addoption(parser):
    parser.addoption(
        '--value-check-runtime',
        dest='value-check-runtime', default='onnxruntime',
        choices=['skip', 'onnxruntime', 'mxnet'], help='select test runtime')
    parser.addoption(
        '--opset-versions', dest='opset-versions', default=None,
        help='select opset versions, select from "min", "latest", '
             'or a list of numbers like "9,10"')


@pytest.fixture(scope='function')
def disable_experimental_warning():
    org_config = chainer.disable_experimental_feature_warning
    chainer.disable_experimental_feature_warning = True
    try:
        yield
    finally:
        chainer.disable_experimental_feature_warning = org_config


@pytest.fixture(scope='function')
def check_model_expect(request):
    selected_runtime = request.config.getoption('value-check-runtime')
    if selected_runtime == 'onnxruntime':
        from onnx_chainer.testing.test_onnxruntime import check_model_expect  # NOQA
        _checker = check_model_expect
    elif selected_runtime == 'mxnet':
        from onnx_chainer.testing.test_mxnet import check_model_expect
        _checker = check_model_expect
    else:
        def empty_func(*args, **kwargs):
            pass
        _checker = empty_func
    return _checker


@pytest.fixture(scope='function')
def target_opsets(request):
    opsets = request.config.getoption('opset-versions')
    min_version = onnx_chainer.MINIMUM_OPSET_VERSION
    max_version = min(
        onnx.defs.onnx_opset_version(), onnx_chainer.MAXIMUM_OPSET_VERSION)
    if opsets is None:
        return list(range(min_version, max_version + 1))
    elif opsets == 'min':
        return [min_version]
    elif opsets == 'latest':
        return [max_version]
    else:
        try:
            versions = [int(i) for i in opsets.split(',')]
        except ValueError:
            raise ValueError('cannot convert {} to versions list'.format(
                opsets))
        return versions
