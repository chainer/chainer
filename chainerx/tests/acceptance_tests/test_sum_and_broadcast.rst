Acceptance tests for sum and broadcast
======================================

>>> import chainerx as chx
>>> import numpy as np

sum
---

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

>>> a.sum()
array(435, shape=(), dtype=int64, device='native:0')
>>> a.sum(keepdims=True)
array([[[435]]], shape=(1, 1, 1), dtype=int64, device='native:0')

>>> a.sum(axis=1)
array([[15, 18, 21, 24, 27],
       [60, 63, 66, 69, 72]], shape=(2, 5), dtype=int64, device='native:0')
>>> a.sum(axis=(-1, 0))
array([ 95, 145, 195], shape=(3,), dtype=int64, device='native:0')
>>> a.sum(axis=(-1, 0), keepdims=True)
array([[[ 95],
        [145],
        [195]]], shape=(1, 3, 1), dtype=int64, device='native:0')

broadcast_to
------------

>>> a = chx.ndarray((3, 1), chx.int64, [1, 2, 3])
>>> a
array([[1],
       [2],
       [3]], shape=(3, 1), dtype=int64, device='native:0')

>>> b = chx.broadcast_to(a, (3, 2))
>>> b
array([[1, 1],
       [2, 2],
       [3, 3]], shape=(3, 2), dtype=int64, device='native:0')
>>> b.strides
(8, 0)

>>> c = chx.broadcast_to(a, (2, 3, 1))
>>> c
array([[[1],
        [2],
        [3]],
<BLANKLINE>
       [[1],
        [2],
        [3]]], shape=(2, 3, 1), dtype=int64, device='native:0')

Broadcast add/mul
-----------------

>>> a = chx.ndarray((3,), chx.int64, [1, 2, 3])
>>> b = chx.ndarray((2, 3), chx.int64, [10, 20, 30, 40, 50, 60])
>>> a + b
array([[11, 22, 33],
       [41, 52, 63]], shape=(2, 3), dtype=int64, device='native:0')
>>> a * b
array([[ 10,  40,  90],
       [ 40, 100, 180]], shape=(2, 3), dtype=int64, device='native:0')

>>> c = chx.ndarray((2, 1), chx.int64, [10, 20])
>>> a + c
array([[11, 12, 13],
       [21, 22, 23]], shape=(2, 3), dtype=int64, device='native:0')
>>> a * c
array([[10, 20, 30],
       [20, 40, 60]], shape=(2, 3), dtype=int64, device='native:0')

Backward
--------

>>> a_np = np.arange(6, dtype=np.float32).reshape(2, 3)
>>> a = chx.array(a_np).require_grad()
>>> a
array([[0., 1., 2.],
       [3., 4., 5.]], shape=(2, 3), dtype=float32, device='native:0', backprop_ids=['<default>'])
>>> b = a.sum(axis=0)
>>> b
array([3., 5., 7.], shape=(3,), dtype=float32, device='native:0', backprop_ids=['<default>'])
>>> c = a * b
>>> c
array([[ 0.,  5., 14.],
       [ 9., 20., 35.]], shape=(2, 3), dtype=float32, device='native:0', backprop_ids=['<default>'])

>>> c.set_grad(chx.ndarray((2, 3), c.dtype, [1, 2, 3, 4, 5, 6]))
>>> chx.backward(c)
>>> a.grad
array([[15., 32., 57.],
       [24., 47., 78.]], shape=(2, 3), dtype=float32, device='native:0')
