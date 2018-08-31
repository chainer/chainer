Acceptance tests for pluggable backend system
=============================================

Using two contexts in one script
--------------------------------

>>> import chainerx as xc

>>> a = xc.ones((3,), xc.float32)
>>> a
array([1., 1., 1.], shape=(3,), dtype=float32, device='native:0')

>>> ctx = xc.Context()
>>> with xc.context_scope(ctx):
...     with xc.device_scope('native'):
...         c = xc.ones((3,), xc.float32)
>>> a.device == c.device
False
