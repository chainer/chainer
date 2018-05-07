Acceptance tests for xchainer.testing.numpy_xchainer_array_equal
================================================================

>>> import xchainer as xc
>>> import numpy as np
>>> import xchainer.testing
>>> import pytest

Arange
------

>>> xc.arange(3)
array([0, 1, 2], shape=(3,), dtype=int64, device='native:0')
>>> xc.arange(start=0.2, stop=3.2)
array([0.2, 1.2, 2.2], shape=(3,), dtype=float64, device='native:0')
>>> xc.arange(start=0.2, stop=1.0, step=0.2)
array([0.2, 0.4, 0.6, 0.8], shape=(4,), dtype=float64, device='native:0')
>>> xc.arange(start=0, stop=1.0, step=0.25, dtype=xc.float32)
array([0.  , 0.25, 0.5 , 0.75], shape=(4,), dtype=float32, device='native:0')

>>> xc.arange(3, device='native:1')
array([0, 1, 2], shape=(3,), dtype=int64, device='native:1')

Squeeze
-------

>>> a = xc.arange(3).reshape(1, 3, 1, 1)
>>> xc.squeeze(a)
array([0, 1, 2], shape=(3,), dtype=int64, device='native:0')
>>> xc.squeeze(a, axis=(2, 3))
array([[0, 1, 2]], shape=(1, 3), dtype=int64, device='native:0')

xchainer.array
--------------

>>> xc.array([0.5, 1, 2])
array([0.5, 1. , 2. ], shape=(3,), dtype=float64, device='native:0')

>>> xc.array(np.arange(3))
array([0, 1, 2], shape=(3,), dtype=int64, device='native:0')

>>> xc.array(xc.arange(3))
array([0, 1, 2], shape=(3,), dtype=int64, device='native:0')


xchainer.testing.assert_array_equal
-----------------------------------

>>> xchainer.testing.assert_array_equal(xc.arange(3), np.arange(3))
>>> xchainer.testing.assert_array_equal(xc.array([0, 1]), np.array([0, 2]))
Traceback (most recent call last):
  ...
AssertionError: 
Arrays are not equal
...


xchainer.testing.numpy_xchainer_array_equal
-------------------------------------------

>>> @xchainer.testing.numpy_xchainer_array_equal()
... @pytest.mark.parametrize('xp', [xc, np])
... def test_array(xp):
...     return xp.array([1, 2, 3], dtype='int32')
>>>
>>> test_array()
