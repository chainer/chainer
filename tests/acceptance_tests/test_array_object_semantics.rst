Acceptance tests for Array Object Semantics
===========================================

>>> import xchainer as xc

>>> a = xc.Array((3,), xc.float32, [1, 2, 3]).require_grad('graph1')
>>> grad = xc.ones((3,), xc.float32)
>>> a.set_grad(grad, 'graph1')
>>> a.get_grad('graph1') is grad
True
