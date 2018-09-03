Acceptance tests for introducing CUDA
=====================================

>>> import chainerx as chx

Create array from python list
-----------------------------

>>> a = xc.ndarray((3,), xc.float32, [1, 2, 3], 'cuda:0')
>>> a.shape
(3,)
>>> a.dtype
dtype.float32
>>> a.device
cuda:0

Create array from numpy ndarray
-------------------------------

>>> import numpy as np
>>> n = np.ones((3,), np.float32, [1, 2, 3])
>>> a = xc.array(n, device='cuda:0')
>>> a.shape
(3,)
>>> a.dtype
dtype.float32
>>> a.device
cuda:0
