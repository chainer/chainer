Acceptance tests for reshape and basic indexing
===============================================

>>> import chainerx as chx
>>> import numpy as np

>>> a_np = np.arange(30).reshape(2, 3, 5)
>>> a = chx.array(a_np)
>>> a
array([[[ 0,  1,  2,  3,  4],
        [ 5,  6,  7,  8,  9],
        [10, 11, 12, 13, 14]],
<BLANKLINE>
       [[15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24],
        [25, 26, 27, 28, 29]]], shape=(2, 3, 5), dtype=int64, device='native:0')

Reshape
-------
>>> a.reshape((30,))
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
       10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
       20, 21, 22, 23, 24, 25, 26, 27, 28, 29], shape=(30,), dtype=int64, device='native:0')
>>> a.reshape((3, 2, 5))
array([[[ 0,  1,  2,  3,  4],
        [ 5,  6,  7,  8,  9]],
<BLANKLINE>
       [[10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19]],
<BLANKLINE>
       [[20, 21, 22, 23, 24],
        [25, 26, 27, 28, 29]]], shape=(3, 2, 5), dtype=int64, device='native:0')
>>> chx.ones((), chx.float32).reshape(1, 1, 1, 1)
array([[[[1.]]]], shape=(1, 1, 1, 1), dtype=float32, device='native:0')
>>> chx.ones((2, 0, 3), chx.float32).reshape(5, 0, 7)
array([], shape=(5, 0, 7), dtype=float32, device='native:0')

Basic indexing
--------------

>>> a[0, 1, -2]
array(8, shape=(), dtype=int64, device='native:0')
>>> a[:]
array([[[ 0,  1,  2,  3,  4],
        [ 5,  6,  7,  8,  9],
        [10, 11, 12, 13, 14]],
<BLANKLINE>
       [[15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24],
        [25, 26, 27, 28, 29]]], shape=(2, 3, 5), dtype=int64, device='native:0')
>>> a[::-1, 1:, -3:5]
array([[[22, 23, 24],
        [27, 28, 29]],
<BLANKLINE>
       [[ 7,  8,  9],
        [12, 13, 14]]], shape=(2, 2, 3), dtype=int64, device='native:0')
>>> a[1, :, -2]
array([18, 23, 28], shape=(3,), dtype=int64, device='native:0')
>>> a[1, chx.newaxis, :, -2, chx.newaxis]
array([[[18],
        [23],
        [28]]], shape=(1, 3, 1), dtype=int64, device='native:0')

Backward
--------

>>> a_np = np.arange(6, dtype=np.float32).reshape(2, 3)
>>> a = chx.array(a_np).require_grad()
>>> a
array([[0., 1., 2.],
       [3., 4., 5.]], shape=(2, 3), dtype=float32, device='native:0', backprop_ids=['<default>'])
>>> b = a.reshape(3, 2)
>>> b
array([[0., 1.],
       [2., 3.],
       [4., 5.]], shape=(3, 2), dtype=float32, device='native:0', backprop_ids=['<default>'])
>>> c = b[1, :]
>>> c
array([2., 3.], shape=(2,), dtype=float32, device='native:0', backprop_ids=['<default>'])

>>> c.set_grad(chx.array(np.array([5, 7], dtype=np.float32)))
>>> chx.backward(c)
>>> a.get_grad()
array([[0., 0., 5.],
       [7., 0., 0.]], shape=(2, 3), dtype=float32, device='native:0')
