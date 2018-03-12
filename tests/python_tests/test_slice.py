import pytest

import xchainer

def test_init():
    # stop
    s = xchainer.Slice(2)
    assert s.start is None
    assert s.stop == 2
    assert s.step == 1

    s = xchainer.Slice(None)
    assert s.start is None
    assert s.stop is None
    assert s.step == 1

    # start, stop
    s = xchainer.Slice(1, 2)
    assert s.start == 1
    assert s.stop == 2
    assert s.step == 1

    s = xchainer.Slice(None, 2)
    assert s.start is None
    assert s.stop == 2
    assert s.step == 1

    s = xchainer.Slice(1, None)
    assert s.start == 1
    assert s.stop is None
    assert s.step == 1

    # start, stop, step
    s = xchainer.Slice(1, 2, 1)
    assert s.start == 1
    assert s.stop == 2
    assert s.step == 1

    s = xchainer.Slice(None, 2, 1)
    assert s.start is None
    assert s.stop == 2
    assert s.step == 1

    s = xchainer.Slice(1, None, 1)
    assert s.start == 1
    assert s.stop is None
    assert s.step == 1

    s = xchainer.Slice(1, 2, None)
    assert s.start == 1
    assert s.stop == 2
    assert s.step == 1


def test_init_by_slice():
    s = xchainer.Slice(slice(1, 2, 1))
    assert s.start == 1
    assert s.stop == 2
    assert s.step == 1

    s = xchainer.Slice(slice(None, 2, 1))
    assert s.start is None
    assert s.stop == 2
    assert s.step == 1

    s = xchainer.Slice(slice(1, None, 1))
    assert s.start == 1
    assert s.stop is None
    assert s.step == 1

    s = xchainer.Slice(slice(1, 2, None))
    assert s.start == 1
    assert s.stop == 2
    assert s.step == 1
