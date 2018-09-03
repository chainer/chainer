Acceptance tests for Backprop
=============================

>>> import chainerx as chx

>>> a = chx.ndarray((2, 3), chx.float32, [0, 1, 2, 3, 4, 5]).require_grad()
>>> b = chx.full_like(a, 2)
>>> a.is_grad_required()
True
>>> b.is_grad_required()
False
>>> y = (a * b) + b
>>> y.is_grad_required()
False
>>> y.is_backprop_required()
True
>>> y.grad = chx.full_like(a, 0.5)
>>> y.is_grad_required()
True
>>> chx.backward(y)

Access gradients through attribute
----------------------------------

>>> a.grad
array([[1., 1., 1.],
       [1., 1., 1.]], shape=(2, 3), dtype=float32, device='native:0')

Access gradients through method
-------------------------------

>>> y.get_grad()
array([[0.5, 0.5, 0.5],
       [0.5, 0.5, 0.5]], shape=(2, 3), dtype=float32, device='native:0')
