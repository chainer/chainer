import pytest


_dtypes_data = [
    {'name': 'bool', 'char': '?', 'itemsize': 1},
    {'name': 'int8', 'char': 'b', 'itemsize': 1},
    {'name': 'int16', 'char': 'h', 'itemsize': 2},
    {'name': 'int32', 'char': 'i', 'itemsize': 4},
    {'name': 'int64', 'char': 'q', 'itemsize': 8},
    {'name': 'uint8', 'char': 'B', 'itemsize': 1},
    {'name': 'float32', 'char': 'f', 'itemsize': 4},
    {'name': 'float64', 'char': 'd', 'itemsize': 8},
]


@pytest.fixture(params=_dtypes_data)
def dtype_data(request):
    return request.param


_devices_data = [
    {'name': 'cpu'},
    {'name': 'cuda'},
]


@pytest.fixture(params=_devices_data)
def device_data(request):
    return request.param


_shapes_data = [
    {'tuple': ()},
    {'tuple': (0,)},
    {'tuple': (1,)},
    {'tuple': (2, 3)},
    {'tuple': (1, 1, 1)},
    {'tuple': (2, 0, 3)},
]


@pytest.fixture(params=_shapes_data)
def shape_data(request):
    return request.param


_scalars_data = [
    {'data': -2},
    {'data': 1},
    {'data': -1.5},
    {'data': 2.3},
    {'data': True},
    {'data': False},
]


@pytest.fixture(params=_scalars_data)
def scalar_data(request):
    return request.param
