Acceptance tests for ndarray object semantics
=============================================

>>> import xchainer as xc

>>> with xc.backprop_scope('bp1') as bp1:
...     a = xc.ndarray((3,), xc.float32, [1, 2, 3]).require_grad(bp1)
...     grad = xc.ones((3,), xc.float32)
...     a.set_grad(grad, bp1)
...     a.get_grad(bp1) is grad
True
