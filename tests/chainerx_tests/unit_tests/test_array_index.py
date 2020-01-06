import pytest
import chainerx


def test_newaxis():
    assert chainerx.newaxis is None


@pytest.mark.parametrize('xp', [chainerx])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('shape, transpose', [
    ((1,), None),
    ((2,), None),
    ((2, 3), None),
    ((2, 3, 4), None),
    ((2, 3, 4, 5), None),
    ((2, 3, 4, 5, 6), None),
    ((2, 3), (0, 1)),
    ((2, 3, 4), (0, 2)),
    ((2, 3, 4, 5), (0, 2)),
    ((2, 3, 4, 5, 6), (1, 3)),
])
def test_array_indexing(xp, device, shape, transpose):
    a = xp.zeros(shape=shape, dtype=chainerx.int8, device=device)
    if transpose:
        a = a.swapaxes(*transpose)
        assert not a.is_contiguous
    a += 1
    assert a.sum() == a.size


@pytest.mark.slow
@pytest.mark.parametrize('xp', [chainerx])
@pytest.mark.parametrize_device(['cuda:0'])
@pytest.mark.parametrize('shape', [
    (64, 32, 6*1024*4),  # Less than 2^32 elems
    (64, 32, 6*1024*512),  # More than 2^32 elems
])
def test_large_array_contiguous_indexing(xp, device, shape):
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
def test_large_array_noncontiguous_indexing(xp, device, shape):
    try:
        a = xp.zeros(shape=shape, dtype=chainerx.int8, device=device)
    except chainerx.ChainerxError as ex:
        assert 'Out of memory' in ex.args
        pytest.skip('Not enough memory to test large indexing')
    a = a.swapaxes(2, 0)
    a += 1
    assert not a.is_contiguous
    assert a.sum() == a.size
