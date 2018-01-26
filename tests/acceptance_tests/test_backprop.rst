>>> import xchainer as xc
>>> a = xc.Array((2, 3), xc.Dtype.float32, [0, 1, 2, 3, 4, 5])
>>> b = xc.full_like(a, 2)
>>> a.requires_grad = True
>>> y = (a * b) + b
>>> y.requires_grad
True
>>> b.requires_grad
False
>>> y.grad = xc.full_like(a, 0.5)
>>> xc.backward(y)
>>> a.grad
array([[1., 1., 1.],
       [1., 1., 1.]], dtype=float32)
>>> y.grad
array([[0.5, 0.5, 0.5],
       [0.5, 0.5, 0.5]], dtype=float32)
