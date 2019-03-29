import copy
import pickle

import pytest

import chainerx


_devices_data = [
    {'index': 0},
    {'index': 1},
]


@pytest.fixture(params=_devices_data)
def device_data1(request):
    return request.param


@pytest.fixture(params=_devices_data)
def device_data2(request):
    return request.param


@pytest.fixture
def device_instance1(request, device_data1):
    return chainerx.get_global_default_context().get_device(
        'native', device_data1['index'])


@pytest.fixture
def device_instance2(request, device_data2):
    return chainerx.get_global_default_context().get_device(
        'native', device_data2['index'])


@pytest.fixture
def cache_restore_device(request):
    device = chainerx.get_default_device()

    def restore_device():
        chainerx.set_default_device(device)
    request.addfinalizer(restore_device)


def test_creation():
    ctx = chainerx.get_global_default_context()
    backend = ctx.get_backend('native')
    device = backend.get_device(0)
    assert device.name == 'native:0'
    assert device.backend is backend
    assert device.context is ctx
    assert device.index == 0

    device = backend.get_device(1)
    assert device.name == 'native:1'
    assert device.backend is backend
    assert device.context is ctx
    assert device.index == 1


def test_synchronize():
    ctx = chainerx.get_global_default_context()
    device = ctx.get_device('native', 0)
    device.synchronize()


@pytest.mark.usefixtures('cache_restore_device')
def test_default_device(device_instance1):
    device = device_instance1
    chainerx.set_default_device(device)
    assert chainerx.get_default_device() is device


@pytest.mark.usefixtures('cache_restore_device')
def test_default_device_with_name(device_instance1):
    device = device_instance1
    chainerx.set_default_device(device.name)
    assert chainerx.get_default_device() is device


@pytest.mark.usefixtures('cache_restore_device')
def test_eq(device_instance1, device_instance2):
    if device_instance1 == device_instance2:
        return

    device1 = device_instance1
    device2 = device_instance2

    device1_1 = device1.backend.get_device(device1.index)
    device1_2 = device1.backend.get_device(device1.index)
    device2_1 = device2.backend.get_device(device2.index)

    assert device1_1 == device1_2
    assert device1_1 != device2_1
    assert not (device1_1 != device1_2)
    assert not (device1_1 == device2_1)


@pytest.mark.usefixtures('cache_restore_device')
def test_using_device(device_instance1, device_instance2):
    if device_instance1 == device_instance2:
        return

    device1 = device_instance1
    device2 = device_instance2

    chainerx.set_default_device(device1)
    with chainerx.using_device(device2) as scope:
        assert chainerx.get_default_device() is device2
        assert scope.device is device2

    scope = chainerx.using_device(device2)
    assert chainerx.get_default_device() == device1
    assert scope.device is device2
    with scope:
        assert chainerx.get_default_device() == device2
        assert scope.device is device2
    assert chainerx.get_default_device() == device1
    assert scope.device is device2


@pytest.mark.usefixtures('cache_restore_device')
def test_using_device_with_name(device_instance1, device_instance2):
    if device_instance1 == device_instance2:
        return

    device1 = device_instance1
    device2 = device_instance2

    chainerx.set_default_device(device1)
    with chainerx.using_device(device2.name) as scope:
        assert chainerx.get_default_device() == device2
        assert scope.device is device2

    with chainerx.using_device(device2.backend.name, device2.index) as scope:
        assert chainerx.get_default_device() == device2
        assert scope.device is device2


# TODO(niboshi): Add pickle test involving context destruction and re-creation
@pytest.mark.parametrize_device(['native:0', 'native:1', 'cuda:0'])
def test_device_pickle(device):
    s = pickle.dumps(device)
    device2 = pickle.loads(s)
    assert device is device2


# TODO(niboshi): Add deepcopy test with arbitrary context
@pytest.mark.parametrize_device(['native:0', 'native:1', 'cuda:0'])
def test_device_deepcopy(device):
    device2 = copy.deepcopy(device)
    assert device is device2
