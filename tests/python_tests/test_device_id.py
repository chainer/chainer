import pytest

import xchainer


_device_ids_data = [
    {'backend': xchainer.NativeBackend(), 'index': 0},
    {'backend': xchainer.NativeBackend(), 'index': 1},
]


@pytest.fixture(params=_device_ids_data)
def device_id_data1(request):
    return request.param


@pytest.fixture(params=_device_ids_data)
def device_id_data2(request):
    return request.param


@pytest.fixture
def device_id_instance1(request, device_id_data1):
    return xchainer.DeviceId(device_id_data1['backend'], device_id_data1['index'])


@pytest.fixture
def device_id_instance2(request, device_id_data2):
    return xchainer.DeviceId(device_id_data2['backend'], device_id_data2['index'])


@pytest.fixture
def cache_restore_device_id(request):
    device_id = xchainer.get_default_device_id()

    def restore_device_id():
        xchainer.set_default_device_id(device_id)
    request.addfinalizer(restore_device_id)


def test_creation():
    backend = xchainer.NativeBackend()
    device_id = xchainer.DeviceId(backend)
    assert device_id.backend is backend
    assert device_id.index == 0

    device_id = xchainer.DeviceId(backend, 1)
    assert device_id.backend is backend
    assert device_id.index == 1


@pytest.mark.usefixtures('cache_restore_device_id')
def test_default_device_id(device_id_instance1):
    device_id = device_id_instance1
    xchainer.set_default_device_id(device_id)
    assert xchainer.get_default_device_id() == xchainer.DeviceId(device_id.backend, device_id.index)


@pytest.mark.usefixtures('cache_restore_device_id')
def test_eq(device_id_instance1, device_id_instance2):
    if device_id_instance1 == device_id_instance2:
        return

    device_id1 = device_id_instance1
    device_id2 = device_id_instance2

    device_id1_1 = xchainer.DeviceId(device_id1.backend, device_id1.index)
    device_id1_2 = xchainer.DeviceId(device_id1.backend, device_id1.index)
    device_id2_1 = xchainer.DeviceId(device_id2.backend, device_id2.index)

    assert device_id1_1 == device_id1_2
    assert device_id1_1 != device_id2_1
    assert not (device_id1_1 != device_id1_2)
    assert not (device_id1_1 == device_id2_1)


@pytest.mark.usefixtures('cache_restore_device_id')
def test_device_scope(device_id_instance1, device_id_instance2):
    if device_id_instance1 == device_id_instance2:
        return

    device_id1 = device_id_instance1
    device_id2 = device_id_instance2

    xchainer.set_default_device_id(device_id1)
    with xchainer.device_scope(device_id2):
        assert xchainer.get_default_device_id() == device_id2

    scope = xchainer.device_scope(device_id2)
    assert xchainer.get_default_device_id() == device_id1
    with scope:
        assert xchainer.get_default_device_id() == device_id2
    assert xchainer.get_default_device_id() == device_id1
