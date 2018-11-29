Acceptance tests for Backprop on GPU
====================================

>>> import chainerx as chx

>>> a = chx.array([[0, 1, 2], [3, 4, 5]], chx.float32, device='cuda:0').require_grad()
>>> b = chx.full_like(a, 2, device='cuda:0')
>>> a.is_grad_required()
True
>>> b.is_grad_required()
False
>>> y = (a * b) + b
>>> y.is_grad_required()
False
>>> y.is_backprop_required()
True
>>> y.set_grad(chx.full_like(a, 0.5, device='cuda:0'))
>>> y.is_grad_required()
True
>>> chx.backward(y)

Access gradients through attribute
----------------------------------

>>> a.grad
array([[1., 1., 1.],
       [1., 1., 1.]], shape=(2, 3), dtype=float32, device='cuda:0')

Access gradients through method
-------------------------------

>>> y.get_grad()
array([[0.5, 0.5, 0.5],
       [0.5, 0.5, 0.5]], shape=(2, 3), dtype=float32, device='cuda:0')
