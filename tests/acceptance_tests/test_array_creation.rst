Acceptance tests for array creation
===================================

>>> import xchainer as xc

Create array from python list
-----------------------------

>>> a = xc.Array((3,), xc.float32, [1, 2, 3])
>>> a.shape
(3,)
>>> a.dtype
dtype.float32

Create array from numpy ndarray
-------------------------------

>>> import numpy as np
>>> n = np.ones((3,), np.float32, [1, 2, 3])
>>> a = xc.Array(n)
>>> a.shape
(3,)
>>> a.dtype
dtype.float32
