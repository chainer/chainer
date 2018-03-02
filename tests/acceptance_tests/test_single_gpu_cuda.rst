Acceptance tests for single GPU
===============================

>>> import xchainer as xc

Addition
--------

>>> a = xc.Array((3,), xc.float32, [1, 2, 3], 'cuda')
>>> b = xc.Array((3,), xc.float32, [4, 5, 6], 'cuda')
>>> y = a + b
>>> y
array([5., 7., 9.], dtype=float32, device='cuda:0')
>>> y += a
>>> y
array([ 6.,  9., 12.], dtype=float32, device='cuda:0')

Multiplication
--------------

>>> a = xc.Array((3,), xc.float32, [1, 2, 3], 'cuda')
>>> b = xc.Array((3,), xc.float32, [4, 5, 6], 'cuda')
>>> y = a * b
>>> y
array([ 4., 10., 18.], dtype=float32, device='cuda:0')
>>> y *= a
>>> y
array([ 4., 20., 54.], dtype=float32, device='cuda:0')

Mixed
-----

>>> a = xc.Array((3,), xc.float32, [1, 2, 3], 'cuda')
>>> b = xc.Array((3,), xc.float32, [4, 5, 6], 'cuda')
>>> c = xc.Array((3,), xc.float32, [7, 8, 9], 'cuda')
>>> y = a + b * c;
>>> y
array([29., 42., 57.], dtype=float32, device='cuda:0')
