import pytest

import chainerx


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
# TODO(niboshi): This test causes segv with CUDA device in certain situations.
# Fix it and remove skip.
@pytest.mark.skip()
def test_device_buffer(device):
    buf = chainerx.testing._DeviceBuffer(
        [1, 2, 3, 4, 5, 6], (2, 3), chainerx.float32, device)
    mv = memoryview(buf)
    assert mv.format == 'f'
    assert mv.itemsize == 4
    assert mv.contiguous
    assert not mv.f_contiguous
    assert not mv.readonly
    assert mv.ndim == 2
    assert mv.shape == (2, 3)
    assert mv.strides == (12, 4)
    assert mv.tolist() == [[1, 2, 3], [4, 5, 6]]
