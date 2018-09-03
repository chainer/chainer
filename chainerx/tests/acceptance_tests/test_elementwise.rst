Acceptance tests for elementwise operations
===========================================

>>> import chainerx as chx
>>> import numpy as np

Array equality
--------------

>>> a = chx.ndarray((2, 3), chx.float32, [1, 2, 3, 4, 5, 6])
>>> b = chx.ndarray((3,), chx.float32, [4, 2, 5])

>>> a == b
array([[False,  True, False],
       [ True, False, False]], shape=(2, 3), dtype=bool, device='native:0')
>>> chx.equal(a, b)
array([[False,  True, False],
       [ True, False, False]], shape=(2, 3), dtype=bool, device='native:0')

Maximum with scalar
-------------------

>>> a = chx.array(np.arange(-3, 4, dtype=np.float32))
>>> a
array([-3., -2., -1.,  0.,  1.,  2.,  3.], shape=(7,), dtype=float32, device='native:0')

>>> chx.maximum(a, -1.0)
array([-1., -1., -1.,  0.,  1.,  2.,  3.], shape=(7,), dtype=float32, device='native:0')
>>> chx.maximum(2.0, a)
array([2., 2., 2., 2., 2., 2., 3.], shape=(7,), dtype=float32, device='native:0')

Multiply with scalar
-------------------

>>> a = chx.array(np.arange(-3, 4, dtype=np.float32))
>>> a
array([-3., -2., -1.,  0.,  1.,  2.,  3.], shape=(7,), dtype=float32, device='native:0')

>>> a * 2.0
array([-6., -4., -2.,  0.,  2.,  4.,  6.], shape=(7,), dtype=float32, device='native:0')
>>> -3.0 * a
array([ 9.,  6.,  3., -0., -3., -6., -9.], shape=(7,), dtype=float32, device='native:0')
>>> chx.multiply(a, 2.0)
array([-6., -4., -2.,  0.,  2.,  4.,  6.], shape=(7,), dtype=float32, device='native:0')
>>> chx.multiply(-3.0, a)
array([ 9.,  6.,  3., -0., -3., -6., -9.], shape=(7,), dtype=float32, device='native:0')

Conversion to Python scalar
---------------------------

>>> a = chx.ndarray((1,), chx.float32, [3.25])
>>> float(a)
3.25
>>> int(a)
3
>>> bool(a)
True
>>> chx.asscalar(a)
3.25

Backward
--------
>>> x = chx.array(np.arange(-3, 3, dtype=np.float32).reshape((2, 3))).require_grad()
>>> x
array([[-3., -2., -1.],
       [ 0.,  1.,  2.]], shape=(2, 3), dtype=float32, device='native:0', backprop_ids=['<default>'])
>>> y = chx.maximum(x, 0) * 2
>>> y
array([[0., 0., 0.],
       [0., 2., 4.]], shape=(2, 3), dtype=float32, device='native:0', backprop_ids=['<default>'])
>>> y.backward()
>>> x.grad
array([[0., 0., 0.],
       [2., 2., 2.]], shape=(2, 3), dtype=float32, device='native:0')
