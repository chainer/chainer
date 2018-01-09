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
