import pytest

import xchainer


_devices_data = [
    {'name': 'cpu', 'backend': xchainer.NativeBackend()},
    {'name': 'cpu2', 'backend': xchainer.NativeBackend()},
]


@pytest.fixture(params=_devices_data)
def device_data1(request):
    return request.param


@pytest.fixture(params=_devices_data)
def device_data2(request):
    return request.param


@pytest.fixture
def device_instance1(request, device_data1):
    return xchainer.Device(device_data1['name'], device_data1['backend'])


@pytest.fixture
def device_instance2(request, device_data2):
    return xchainer.Device(device_data2['name'], device_data2['backend'])


@pytest.fixture
def cache_restore_device(request):
    device = xchainer.get_current_device()

    def restore_device():
        xchainer.set_current_device(device)
    request.addfinalizer(restore_device)


def test_creation():
    backend = xchainer.NativeBackend()
    device = xchainer.Device('cpu', backend)
    assert device.name == 'cpu'
    assert device.backend is backend


@pytest.mark.usefixtures('cache_restore_device')
def test_current_device(device_instance1):
    device = device_instance1
    xchainer.set_current_device(device)
    assert xchainer.get_current_device() == xchainer.Device(device.name, device.backend)


@pytest.mark.usefixtures('cache_restore_device')
def test_eq(device_instance1, device_instance2):
    if device_instance1 == device_instance2:
        return

    device1 = device_instance1
    device2 = device_instance2

    device1_1 = xchainer.Device(device1.name, device1.backend)
    device1_2 = xchainer.Device(device1.name, device1.backend)
    device2_1 = xchainer.Device(device2.name, device2.backend)

    assert device1_1 == device1_2
    assert device1_1 != device2_1
    assert not (device1_1 != device1_2)
    assert not (device1_1 == device2_1)


@pytest.mark.usefixtures('cache_restore_device')
def test_device_scope(device_instance1, device_instance2):
    if device_instance1 == device_instance2:
        return

    device1 = device_instance1
    device2 = device_instance2

    xchainer.set_current_device(device1)
    with xchainer.device_scope(device2):
        assert xchainer.get_current_device() == device2

    scope = xchainer.device_scope(device2)
    assert xchainer.get_current_device() == device1
    with scope:
        assert xchainer.get_current_device() == device2
    assert xchainer.get_current_device() == device1


def test_init_invalid_length():
    with pytest.raises(xchainer.DeviceError):
        xchainer.Device('a' * 8, xchainer.NativeBackend())  # too long device name
