import pytest

import xchainer


_strides_data = [
    {'tuple': ()},
    {'tuple': (0,)},
    {'tuple': (4,)},
    {'tuple': (16, 4)},
    {'tuple': (48, 16, 4)},
]


@pytest.fixture(params=_strides_data)
def strides_data(request):
    return request.param


@pytest.fixture
def strides_init_inputs(request, strides_data):
    return strides_data['tuple']


def test_attr(strides_init_inputs):
    tup = strides_init_inputs
    strides = xchainer.Strides(tup)

    assert strides.ndim == len(tup)
    assert str(strides) == str(tup)


def test_eq(strides_init_inputs):
    tup = strides_init_inputs
    strides = xchainer.Strides(tup)

    # equality
    assert strides == xchainer.Strides(tup)
    assert strides == tup
    assert tup == strides

    # inequality
    assert strides != xchainer.Strides(tup + (1,))
    assert strides != tup + (1,)
    assert tup + (1,) != strides
    if tup != ():
        assert strides != tuple(['a' for _ in tup])
        # Note: this behavior is different from NumPy
        assert strides != tuple([float(d) for d in tup])
