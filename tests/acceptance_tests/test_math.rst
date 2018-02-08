Acceptance tests for basic math
===============================

>>> import xchainer as xc

Addition
--------

>>> a = xc.Array((3,), xc.Dtype.float32, [1, 2, 3])
>>> b = xc.Array((3,), xc.Dtype.float32, [4, 5, 6])
>>> y = a + b
>>> y
array([5., 7., 9.], dtype=float32, device='cpu')
>>> y += a
>>> y
array([ 6.,  9., 12.], dtype=float32, device='cpu')


Multiplication
--------------

>>> a = xc.Array((3,), xc.Dtype.float32, [1, 2, 3])
>>> b = xc.Array((3,), xc.Dtype.float32, [4, 5, 6])
>>> y = a * b
>>> y
array([ 4., 10., 18.], dtype=float32, device='cpu')
>>> y *= a
>>> y
array([ 4., 20., 54.], dtype=float32, device='cpu')


Mixed
-----

>>> a = xc.Array((3,), xc.Dtype.float32, [1, 2, 3])
>>> b = xc.Array((3,), xc.Dtype.float32, [4, 5, 6])
>>> c = xc.Array((3,), xc.Dtype.float32, [7, 8, 9])
>>> y = a + b * c;
>>> y
array([29., 42., 57.], dtype=float32, device='cpu')
