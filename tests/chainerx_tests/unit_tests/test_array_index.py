import pytest
import chainerx


def test_newaxis():
    assert chainerx.newaxis is None


@pytest.mark.slow
@pytest.mark.parametrize('xp', [chainerx])
@pytest.mark.parametrize_device(['cuda:0'])
@pytest.mark.parametrize('shape', [
    (64, 32, 6*1024*4),  # Less than 2^32 elems
    (64, 32, 6*1024*512),  # More than 2^32 elems
])
def test_array_contiguous_indexing(xp, device, shape):
    try:
        a = xp.zeros(shape=shape, dtype=chainerx.int8, device=device)
    except chainerx.ChainerxError as ex:
        assert 'Out of memory' in ex.args
        pytest.skip('Not enough memory to test large indexing')
    a += 1
    assert a.is_contiguous
    assert a.sum() == a.size


@pytest.mark.slow
@pytest.mark.parametrize('xp', [chainerx])
@pytest.mark.parametrize_device(['cuda:0'])
@pytest.mark.parametrize('shape', [
    (64, 32, 6*1024*4),  # Less than 2^32 elems
    (64, 32, 6*1024*512)  # More than 2^32 elems
])
def test_array_noncontiguous_indexing(xp, device, shape):
    try:
        a = xp.zeros(shape=shape, dtype=chainerx.int8, device=device)
    except chainerx.ChainerxError as ex:
        assert 'Out of memory' in ex.args
        pytest.skip('Not enough memory to test large indexing')
    a = a.swapaxes(2, 0)
    a += 1
    assert not a.is_contiguous
    assert a.sum() == a.size
