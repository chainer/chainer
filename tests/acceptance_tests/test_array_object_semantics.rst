Acceptance tests for ndarray object semantics
=============================================

>>> import xchainer as xc

>>> with xc.graph_scope('graph1') as graph1:
...     a = xc.ndarray((3,), xc.float32, [1, 2, 3]).require_grad(graph1)
...     grad = xc.ones((3,), xc.float32)
...     a.set_grad(grad, graph1)
...     a.get_grad(graph1) is grad
True
