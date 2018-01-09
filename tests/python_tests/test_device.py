import pytest

import xchainer


_devices_data = [
    {'name': 'cpu'},
    {'name': 'cuda'},
]


@pytest.fixture(params=_devices_data)
def device_data1(request):
    return request.param


@pytest.fixture(params=_devices_data)
def device_data2(request):
    return request.param


@pytest.fixture
def device_init_inputs1(request, device_data1):
    return device_data1['name']


@pytest.fixture
def device_init_inputs2(request, device_data2):
    return device_data2['name']


@pytest.fixture
def cache_restore_device(request):
    device = xchainer.get_current_device()

    def restore_device():
        xchainer.set_current_device(device)
    request.addfinalizer(restore_device)


@pytest.mark.usefixtures('cache_restore_device')
def test_current_device(device_init_inputs1):
    name = device_init_inputs1
    xchainer.set_current_device(name)
    assert xchainer.get_current_device() == xchainer.Device(name)


@pytest.mark.usefixtures('cache_restore_device')
def test_eq(device_init_inputs1, device_init_inputs2):
    if device_init_inputs1 == device_init_inputs2:
        return

    name1 = device_init_inputs1
    name2 = device_init_inputs2

    device1_1 = xchainer.Device(name1)
    device1_2 = xchainer.Device(name1)
    device2 = xchainer.Device(name2)

    assert device1_1 == device1_2
    assert device1_1 != device2
    assert not (device1_1 != device1_2)
    assert not (device1_1 == device2)


@pytest.mark.usefixtures('cache_restore_device')
def test_device_scope(device_init_inputs1, device_init_inputs2):
    if device_init_inputs1 == device_init_inputs2:
        return

    name1 = device_init_inputs1
    name2 = device_init_inputs2

    device1 = xchainer.Device(name1)
    device2 = xchainer.Device(name2)

    xchainer.set_current_device(name1)
    with xchainer.device_scope(name2):
        assert xchainer.get_current_device() == device2

    scope = xchainer.device_scope(name2)
    assert xchainer.get_current_device() == device1
    with scope:
        assert xchainer.get_current_device() == device2
    assert xchainer.get_current_device() == device1


def test_init_invalid_lengt():
    with pytest.raises(xchainer.DeviceError):
        xchainer.Device('a' * 8)  # too long device name


def test_set_current_invalid_name():
    with pytest.raises(xchainer.DeviceError):
        xchainer.set_current_device('invalid_device')
