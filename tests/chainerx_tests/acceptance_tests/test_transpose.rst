Acceptance tests for Transpose
==============================

>>> import chainerx as chx

Using the method
----------------

>>> a = chx.array([[1, 2, 3], [4, 5, 6]], chx.float32)
>>> b = a.transpose()
>>> b
array([[1., 4.],
       [2., 5.],
       [3., 6.]], shape=(3, 2), dtype=float32, device='native:0')

Using the T alias
-----------------

>>> a = chx.array([[[1, 2, 3]], [[4, 5, 6]]], chx.float32)
>>> b = a.T
>>> b
array([[[1., 4.]],
<BLANKLINE>
       [[2., 5.]],
<BLANKLINE>
       [[3., 6.]]], shape=(3, 1, 2), dtype=float32, device='native:0')

Copy a non-contiguous array
---------------------------

>>> a = chx.array([[1, 2, 3], [4, 5, 6]], chx.float32)
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

>>> a = chx.array([[1, 2, 3], [4, 5, 6]], chx.float32).require_grad()
>>> b = a.transpose()
>>> b.is_contiguous
False

>>> c = chx.array([[-2, 1], [3, -1], [1, 0]], chx.float32)
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
       [ 3.,  0.]], shape=(3, 2), dtype=float32, device='native:0', backprop_ids=['<default>'])
>>> y.is_contiguous
True
>>> y.set_grad(chx.full_like(y, 0.5))
>>> chx.backward(y)

>>> a.get_grad()
array([[-1. ,  1.5,  0.5],
       [ 0.5, -0.5,  0. ]], shape=(2, 3), dtype=float32, device='native:0')
