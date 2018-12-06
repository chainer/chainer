Acceptance tests for array creation
===================================

>>> import chainerx as chx

Create uninitialized array
--------------------------

>>> a = chx.ndarray((3,), chx.float32)
>>> a.shape
(3,)
>>> a.dtype
dtype('float32')

Create array from python list
-----------------------------

>>> a = chx.array([1, 2, 3], chx.float32)
>>> a.shape
(3,)
>>> a.dtype
dtype('float32')

Create array from numpy ndarray
-------------------------------

>>> import numpy as np
>>> n = np.ones((3,), np.float32)
>>> a = chx.array(n)
>>> a.shape
(3,)
>>> a.dtype
dtype('float32')
