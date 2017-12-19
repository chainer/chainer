import pytest

import xchainer


def test_core():
    assert xchainer.__name__ == 'xchainer'


def test_dtype():
    assert xchainer.Dtype("bool") == xchainer.bool
    assert xchainer.Dtype("int8") == xchainer.int8
    assert xchainer.Dtype("int16") == xchainer.int16
    assert xchainer.Dtype("int32") == xchainer.int32
    assert xchainer.Dtype("int64") == xchainer.int64
    assert xchainer.Dtype("uint8") == xchainer.uint8
    assert xchainer.Dtype("float32") == xchainer.float32
    assert xchainer.Dtype("float64") == xchainer.float64
    assert xchainer.Dtype("?") == xchainer.bool
    assert xchainer.Dtype("b") == xchainer.int8
    assert xchainer.Dtype("h") == xchainer.int16
    assert xchainer.Dtype("i") == xchainer.int32
    assert xchainer.Dtype("q") == xchainer.int64
    assert xchainer.Dtype("B") == xchainer.uint8
    assert xchainer.Dtype("f") == xchainer.float32
    assert xchainer.Dtype("d") == xchainer.float64
    assert xchainer.bool.name == "bool"
    assert xchainer.int8.name == "int8"
    assert xchainer.int16.name == "int16"
    assert xchainer.int32.name == "int32"
    assert xchainer.int64.name == "int64"
    assert xchainer.uint8.name == "uint8"
    assert xchainer.float32.name == "float32"
    assert xchainer.float64.name == "float64"
    assert xchainer.bool.char == "?"
    assert xchainer.int8.char == "b"
    assert xchainer.int16.char == "h"
    assert xchainer.int32.char == "i"
    assert xchainer.int64.char == "q"
    assert xchainer.uint8.char == "B"
    assert xchainer.float32.char == "f"
    assert xchainer.float64.char == "d"
    assert xchainer.bool.itemsize == 1
    assert xchainer.int8.itemsize == 1
    assert xchainer.int16.itemsize == 2
    assert xchainer.int32.itemsize == 4
    assert xchainer.int64.itemsize == 8
    assert xchainer.uint8.itemsize == 1
    assert xchainer.float32.itemsize == 4
    assert xchainer.float64.itemsize == 8


@pytest.fixture(params=[-2, 1, -1.5, 2.3, True, False])
def scalar_data(request):
    return request.param


class TestScalar(object):

    def assert_casts(self, scalar, scalar_data):
        assert bool(scalar) == bool(scalar_data)
        assert int(scalar) == int(scalar_data)
        assert float(scalar) == float(scalar_data)

    def check_cast(self, scalar_data):
        scalar = xchainer.Scalar(scalar_data)
        self.assert_casts(scalar, scalar_data)

    # operator+()
    def check_pos_cast(self, scalar_data):
        scalar = xchainer.Scalar(scalar_data)
        self.assert_casts(+scalar, +scalar_data)

    # operator-()
    def check_neg_cast(self, scalar_data):
        scalar = xchainer.Scalar(scalar_data)

        if isinstance(scalar_data, bool):
            with pytest.raises(xchainer.DtypeError):
                -scalar  # Should not be able to negate bool
        else:
            self.assert_casts(-scalar, -scalar_data)

    def test_cast(self, scalar_data):
        self.check_cast(scalar_data)
        self.check_pos_cast(scalar_data)
        self.check_neg_cast(scalar_data)

    def test_repr(self, scalar_data):
        assert repr(xchainer.Scalar(scalar_data)) == repr(scalar_data)
        assert str(xchainer.Scalar(scalar_data)) == str(scalar_data)

    def test_dtype(self, scalar_data):
        scalar = xchainer.Scalar(scalar_data)
        if isinstance(scalar_data, bool):
            assert scalar.dtype == xchainer.bool
        elif isinstance(scalar_data, int):
            assert scalar.dtype == xchainer.int64
        elif isinstance(scalar_data, float):
            assert scalar.dtype == xchainer.float64
        else:
            assert False

    def test_init_invalid(self):
        with pytest.raises(TypeError):
            xchainer.Scalar("1")
