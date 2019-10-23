import pytest
from chainermn import testing
import chainer.testing.attr


# use_chainerx, expected
test_data_cpu = [
    (False, "@numpy"),
    (True, "native:0"),
]

# gpu_id, use_chainerx, expected
test_data_gpu = [
    (0, False, "@cupy:0"),
    (1, False, "@cupy:1"),
    (0, True, "cuda:0"),
    (1, True, "cuda:1"),
]


@pytest.mark.parametrize("use_chainerx,expected", test_data_cpu)
def test_get_device_cpu(use_chainerx, expected):
    device = testing.get_device(use_chainerx=use_chainerx)
    assert device.name == expected


@chainer.testing.attr.gpu
@pytest.mark.parametrize("gpu_id,use_chainerx,expected", test_data_gpu)
def test_get_device(gpu_id, use_chainerx, expected):
    device = testing.get_device(gpu_id, use_chainerx)
    assert device.name == expected
