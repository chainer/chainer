Acceptance tests for pluggable backend system
=============================================

Using two contexts in one script
--------------------------------

>>> import chainerx as chx

>>> a = chx.ones((3,), chx.float32)
>>> a
array([1., 1., 1.], shape=(3,), dtype=float32, device='native:0')

>>> ctx = chx.Context()
>>> with chx.context_scope(ctx):
...     with chx.device_scope('native'):
...         c = chx.ones((3,), chx.float32)
>>> a.device == c.device
False
