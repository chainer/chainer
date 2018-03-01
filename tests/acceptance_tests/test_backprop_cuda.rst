Acceptance tests for Backprop on GPU
====================================

>>> import xchainer as xc

>>> a = xc.Array((2, 3), xc.Dtype.float32, [0, 1, 2, 3, 4, 5], device='cuda:0').require_grad()
>>> b = xc.full_like(a, 2, device='cuda:0')
>>> y = (a * b) + b
>>> y.is_grad_required()
True
>>> b.is_grad_required()
False
>>> y.set_grad(xc.full_like(a, 0.5, device='cuda:0'))
>>> xc.backward(y)
>>> a.get_grad()
array([[1., 1., 1.],
       [1., 1., 1.]], dtype=float32, device='cuda:0')
>>> y.get_grad()
array([[0.5, 0.5, 0.5],
       [0.5, 0.5, 0.5]], dtype=float32, device='cuda:0')
