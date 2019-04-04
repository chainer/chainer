import pytest

import chainerx.testing

from chainerx_tests import cuda_utils


def pytest_configure(config):
    _register_cuda_marker(config)


def pytest_runtest_setup(item):
    _setup_cuda_marker(item)


def pytest_runtest_teardown(item, nextitem):
    current_device = cuda_utils.get_current_device()
    assert current_device is None or current_device == 0


def pytest_generate_tests(metafunc):
    marker = [
        m for m in metafunc.definition.iter_markers()
        if m.name == 'parametrize_device']
    if len(marker) > 0:
        marker, = marker  # asserts len == 1
        device_names, = marker.args
        metafunc.parametrize('device', device_names, indirect=True)


def _register_cuda_marker(config):
    config.addinivalue_line(
        'markers',
        'cuda(num=1): mark tests needing the specified number of NVIDIA GPUs.')


def _setup_cuda_marker(item):
    """Pytest marker to indicate number of NVIDIA GPUs required to run the test.

    Tests can be annotated with this decorator (e.g., ``@pytest.mark.cuda``) to
    declare that one NVIDIA GPU is required to run.

    Tests can also be annotated as ``@pytest.mark.cuda(2)`` to declare number
    of NVIDIA GPUs required to run. When running tests, if
    ``CHAINERX_TEST_CUDA_DEVICE_LIMIT`` environment variable is set to value
    greater than or equals to 0, test cases that require GPUs more than the
    limit will be skipped.
    """

    cuda_marker = [m for m in item.iter_markers() if m.name == 'cuda']
    if len(cuda_marker) > 0:
        cuda_marker, = cuda_marker  # asserts len == 1
        required_num = cuda_marker.args[0] if cuda_marker.args else 1
        if cuda_utils.get_cuda_limit() < required_num:
            pytest.skip('{} NVIDIA GPUs required'.format(required_num))


def _get_required_cuda_devices_from_device_name(device_name):
    # Returns the number of required CUDA devices to run a test, given a
    # device name. If the device is non-CUDA device, 0 is returned.
    s = device_name.split(':')
    assert len(s) == 2
    if s[0] != 'cuda':
        return 0
    return int(s[1]) + 1


@pytest.fixture
def device(request):
    # A fixture to wrap a test with a device scope, given a device name.
    # Device instance is passed to the test.

    device_name = request.param

    # Skip if the device is CUDA device and there's no sufficient CUDA devices.
    cuda_device_count = _get_required_cuda_devices_from_device_name(
        device_name)
    if cuda_device_count > cuda_utils.get_cuda_limit():
        pytest.skip()

    device = chainerx.get_device(device_name)
    device_scope = chainerx.using_device(device)

    def finalize():
        device_scope.__exit__()

    request.addfinalizer(finalize)
    device_scope.__enter__()
    return device


@pytest.fixture(params=chainerx.testing.all_dtypes)
def dtype(request):
    return request.param


@pytest.fixture(params=chainerx.testing.float_dtypes)
def float_dtype(request):
    return request.param


@pytest.fixture(params=chainerx.testing.signed_dtypes)
def signed_dtype(request):
    return request.param


@pytest.fixture(params=chainerx.testing.numeric_dtypes)
def numeric_dtype(request):
    return request.param


@pytest.fixture(params=[True, False])
def is_module(request):
    return request.param


_shapes = [
    (),
    (0,),
    (1,),
    (2, 3),
    (1, 1, 1),
    (2, 0, 3),
]

_shapes_as_tuple_or_int = _shapes + [0, 1, 5]


@pytest.fixture(params=_shapes)
def shape(request):
    return request.param


@pytest.fixture(params=_shapes_as_tuple_or_int)
def shape_as_tuple_or_int(request):
    return request.param
