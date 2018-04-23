Acceptance tests for array creation
===================================

>>> import xchainer as xc

Create array from python list
-----------------------------

>>> a = xc.ndarray((3,), xc.float32, [1, 2, 3])
>>> a.shape
(3,)
>>> a.dtype
dtype.float32

Create array from numpy ndarray
-------------------------------

>>> import numpy as np
>>> n = np.ones((3,), np.float32, [1, 2, 3])
>>> a = xc.array(n)
>>> a.shape
(3,)
>>> a.dtype
dtype.float32
