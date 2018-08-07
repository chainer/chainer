Acceptance tests for Transpose
==============================

>>> import xchainer as xc

Using the method
----------------

>>> a = xc.ndarray((2, 3), xc.float32, [1, 2, 3, 4, 5, 6])
>>> b = a.transpose()
>>> b
array([[1., 4.],
       [2., 5.],
       [3., 6.]], shape=(3, 2), dtype=float32, device='native:0')

Using the T alias
-----------------

>>> a = xc.ndarray((2, 1, 3), xc.float32, [1, 2, 3, 4, 5, 6])
>>> b = a.T
>>> b
array([[[1., 4.]],
<BLANKLINE>
       [[2., 5.]],
<BLANKLINE>
       [[3., 6.]]], shape=(3, 1, 2), dtype=float32, device='native:0')

Copy a non-contiguous array
---------------------------

>>> a = xc.ndarray((2, 3), xc.float32, [1, 2, 3, 4, 5, 6])
>>> b = a.transpose()
>>> b.is_contiguous
False

>>> c = b.copy()
>>> c
array([[1., 4.],
       [2., 5.],
       [3., 6.]], shape=(3, 2), dtype=float32, device='native:0')
>>> c.is_contiguous
True

Mixed contiguity arithmetics and Backprop
-----------------------------------------

>>> a = xc.ndarray((2, 3), xc.float32, [1, 2, 3, 4, 5, 6]).require_grad()
>>> b = a.transpose()
>>> b.is_contiguous
False

>>> c = xc.ndarray((3, 2), xc.float32, [-2, 1, 3, -1, 1, 0])
>>> c
array([[-2.,  1.],
       [ 3., -1.],
       [ 1.,  0.]], shape=(3, 2), dtype=float32, device='native:0')
>>> c.is_contiguous
True

>>> y = b * c
>>> y
array([[-2.,  4.],
       [ 6., -5.],
       [ 3.,  0.]], shape=(3, 2), dtype=float32, device='native:0', backprop_ids=['0'])
>>> y.is_contiguous
True
>>> y.set_grad(xc.full_like(y, 0.5))
>>> xc.backward(y)

>>> a.get_grad()
array([[-1. ,  1.5,  0.5],
       [ 0.5, -0.5,  0. ]], shape=(2, 3), dtype=float32, device='native:0')
