Acceptance tests for chainerx.testing.numpy_chainerx_array_equal
================================================================

>>> import chainerx as chx
>>> import numpy as np
>>> import chainerx.testing
>>> import pytest

Arange
------

>>> chx.arange(3)
array([0, 1, 2], shape=(3,), dtype=int32, device='native:0')
>>> chx.arange(start=0.5, stop=3.5)
array([0.5, 1.5, 2.5], shape=(3,), dtype=float32, device='native:0')
>>> chx.arange(start=0.25, stop=1.25, step=0.25)
array([0.25, 0.5 , 0.75, 1.  ], shape=(4,), dtype=float32, device='native:0')
>>> chx.arange(start=0, stop=1.0, step=0.2, dtype=chx.float64)
array([0. , 0.2, 0.4, 0.6, 0.8], shape=(5,), dtype=float64, device='native:0')

>>> chx.arange(3, device='native:1')
array([0, 1, 2], shape=(3,), dtype=int32, device='native:1')

Squeeze
-------

>>> a = chx.arange(3).reshape(1, 3, 1, 1)
>>> chx.squeeze(a)
array([0, 1, 2], shape=(3,), dtype=int32, device='native:0')
>>> chx.squeeze(a, axis=(2, 3))
array([[0, 1, 2]], shape=(1, 3), dtype=int32, device='native:0')

chainerx.array
--------------

>>> chx.array([0.5, 1, 2])
array([0.5, 1. , 2. ], shape=(3,), dtype=float64, device='native:0')

>>> chx.array(np.arange(3))
array([0, 1, 2], shape=(3,), dtype=int64, device='native:0')

>>> chx.array(chx.arange(3))
array([0, 1, 2], shape=(3,), dtype=int32, device='native:0')


chainerx.testing.assert_array_equal
-----------------------------------

>>> chainerx.testing.assert_array_equal(chx.arange(3), np.arange(3))
>>> chainerx.testing.assert_array_equal(chx.array([0, 1]), np.array([0, 2]))
Traceback (most recent call last):
  ...
AssertionError: 
Arrays are not equal
...


chainerx.testing.numpy_chainerx_array_equal
-------------------------------------------

>>> @chainerx.testing.numpy_chainerx_array_equal()
... @pytest.mark.parametrize('xp', [chx, np])
... def test_array(xp):
...     return xp.array([1, 2, 3], dtype='int32')
>>>
>>> test_array()
