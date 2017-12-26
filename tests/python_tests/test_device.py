import pytest

import xchainer


@pytest.fixture
def inputs(request, device_data):
    return device_data['name']


@pytest.fixture
def inputs1(request, inputs):
    return inputs


@pytest.fixture
def inputs2(request, inputs):
    return inputs


@pytest.fixture
def cache_restore_device(request):
    device = xchainer.get_current_device()

    def restore_device():
        xchainer.set_current_device(device)
    request.addfinalizer(restore_device)


@pytest.mark.usefixtures('cache_restore_device')
def test_eq(inputs1, inputs2):
    if inputs1 == inputs2:
        return

    name1 = inputs1
    name2 = inputs2

    device1_1 = xchainer.Device(name1)
    device1_2 = xchainer.Device(name1)
    device2 = xchainer.Device(name2)

    assert device1_1 == device1_2
    assert device1_1 != device2
    assert not (device1_1 != device1_2)
    assert not (device1_1 == device2)


@pytest.mark.usefixtures('cache_restore_device')
def test_current_device(inputs):
    name = inputs
    xchainer.set_current_device(name)
    assert xchainer.get_current_device() == xchainer.Device(name)


@pytest.mark.usefixtures('cache_restore_device')
def test_device_scope(inputs1, inputs2):
    if inputs1 == inputs2:
        return

    name1 = inputs1
    name2 = inputs2

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
