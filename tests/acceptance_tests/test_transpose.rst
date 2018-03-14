Acceptance tests for Transpose
==============================

>>> import xchainer as xc

Using the method
----------------

>>> a = xc.Array((2, 3), xc.float32, [1, 2, 3, 4, 5, 6])
>>> b = a.transpose()
>>> b
array([[1., 4.],
       [2., 5.],
       [3., 6.]], shape=(3, 2), dtype=float32, device='native:0')

Using the T alias
-----------------

>>> a = xc.Array((2, 1, 3), xc.float32, [1, 2, 3, 4, 5, 6])
>>> b = a.T
>>> b
array([[[1., 4.]],
<BLANKLINE>
       [[2., 5.]],
<BLANKLINE>
       [[3., 6.]]], shape=(3, 1, 2), dtype=float32, device='native:0')
