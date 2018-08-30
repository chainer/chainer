Acceptance tests for single GPU
===============================

>>> import xchainer as xc

Addition
--------

>>> a = xc.ndarray((3,), xc.float32, [1, 2, 3], 'cuda')
>>> b = xc.ndarray((3,), xc.float32, [4, 5, 6], 'cuda')
>>> y = a + b
>>> y
array([5., 7., 9.], shape=(3,), dtype=float32, device='cuda:0')
>>> y += a
>>> y
array([ 6.,  9., 12.], shape=(3,), dtype=float32, device='cuda:0')

Multiplication
--------------

>>> a = xc.ndarray((3,), xc.float32, [1, 2, 3], 'cuda')
>>> b = xc.ndarray((3,), xc.float32, [4, 5, 6], 'cuda')
>>> y = a * b
>>> y
array([ 4., 10., 18.], shape=(3,), dtype=float32, device='cuda:0')
>>> y *= a
>>> y
array([ 4., 20., 54.], shape=(3,), dtype=float32, device='cuda:0')

Mixed
-----

>>> a = xc.ndarray((3,), xc.float32, [1, 2, 3], 'cuda')
>>> b = xc.ndarray((3,), xc.float32, [4, 5, 6], 'cuda')
>>> c = xc.ndarray((3,), xc.float32, [7, 8, 9], 'cuda')
>>> y = a + b * c;
>>> y
array([29., 42., 57.], shape=(3,), dtype=float32, device='cuda:0')

ndarray.copy() copies the array on the same device
--------------------------------------------------

>>> with xc.device_scope('native'):  # Allocate arrays on CPU
...     a = xc.ndarray((2, 3), xc.float32, [0, 1, 2, 3, 4, 5]).require_grad()
...
>>> a.device
native:0
>>> with xc.device_scope('cuda'):
...     a = a.copy()
...
>>> a.device
native:0
