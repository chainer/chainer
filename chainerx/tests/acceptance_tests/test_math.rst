Acceptance tests for basic math
===============================

>>> import chainerx as chx

Addition
--------

>>> a = chx.ndarray((3,), chx.float32, [1, 2, 3])
>>> b = chx.ndarray((3,), chx.float32, [4, 5, 6])
>>> y = a + b
>>> y
array([5., 7., 9.], shape=(3,), dtype=float32, device='native:0')
>>> y += a
>>> y
array([ 6.,  9., 12.], shape=(3,), dtype=float32, device='native:0')

Multiplication
--------------

>>> a = chx.ndarray((3,), chx.float32, [1, 2, 3])
>>> b = chx.ndarray((3,), chx.float32, [4, 5, 6])
>>> y = a * b
>>> y
array([ 4., 10., 18.], shape=(3,), dtype=float32, device='native:0')
>>> y *= a
>>> y
array([ 4., 20., 54.], shape=(3,), dtype=float32, device='native:0')

Mixed
-----

>>> a = chx.ndarray((3,), chx.float32, [1, 2, 3])
>>> b = chx.ndarray((3,), chx.float32, [4, 5, 6])
>>> c = chx.ndarray((3,), chx.float32, [7, 8, 9])
>>> y = a + b * c
>>> y
array([29., 42., 57.], shape=(3,), dtype=float32, device='native:0')
