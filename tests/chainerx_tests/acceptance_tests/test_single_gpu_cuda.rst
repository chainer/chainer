Acceptance tests for single GPU
===============================

>>> import chainerx as chx

Addition
--------

>>> a = chx.array([1, 2, 3], chx.float32, device='cuda')
>>> b = chx.array([4, 5, 6], chx.float32, device='cuda')
>>> y = a + b
>>> y
array([5., 7., 9.], shape=(3,), dtype=float32, device='cuda:0')
>>> y += a
>>> y
array([ 6.,  9., 12.], shape=(3,), dtype=float32, device='cuda:0')

Multiplication
--------------

>>> a = chx.array([1, 2, 3], chx.float32, device='cuda')
>>> b = chx.array([4, 5, 6], chx.float32, device='cuda')
>>> y = a * b
>>> y
array([ 4., 10., 18.], shape=(3,), dtype=float32, device='cuda:0')
>>> y *= a
>>> y
array([ 4., 20., 54.], shape=(3,), dtype=float32, device='cuda:0')

Mixed
-----

>>> a = chx.array([1, 2, 3], chx.float32, device='cuda')
>>> b = chx.array([4, 5, 6], chx.float32, device='cuda')
>>> c = chx.array([7, 8, 9], chx.float32, device='cuda')
>>> y = a + b * c;
>>> y
array([29., 42., 57.], shape=(3,), dtype=float32, device='cuda:0')

ndarray.copy() copies the array on the same device
--------------------------------------------------

>>> with chx.device_scope('native'):  # Allocate arrays on CPU
...     a = chx.array([[0, 1, 2], [3, 4, 5]], chx.float32).require_grad()
...
>>> a.device
native:0
>>> with chx.device_scope('cuda'):
...     a = a.copy()
...
>>> a.device
native:0
