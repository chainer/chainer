Acceptance tests for Backprop on GPU
====================================

>>> import xchainer as xc

>>> a = xc.ndarray((2, 3), xc.float32, [0, 1, 2, 3, 4, 5], device='cuda:0').require_grad()
>>> b = xc.full_like(a, 2, device='cuda:0')
>>> y = (a * b) + b
>>> y.is_backprop_required()
True
>>> b.is_backprop_required()
False
>>> y.set_grad(xc.full_like(a, 0.5, device='cuda:0'))
>>> xc.backward(y)

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
