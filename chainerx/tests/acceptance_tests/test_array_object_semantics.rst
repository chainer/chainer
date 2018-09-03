Acceptance tests for ndarray object semantics
=============================================

>>> import chainerx as chx

>>> with chx.backprop_scope('bp1') as bp1:
...     a = chx.ndarray((3,), chx.float32, [1, 2, 3]).require_grad(bp1)
...     grad = chx.ones((3,), chx.float32)
...     a.set_grad(grad, bp1)
...     a.get_grad(bp1) is grad
True
