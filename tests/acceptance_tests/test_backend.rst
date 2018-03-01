Acceptance tests for pluggable backend system
=============================================

Using two backends in one script
--------------------------------

>>> import xchainer as xc

>>> a = xc.ones((3,), xc.float32)
>>> a
array([1., 1., 1.], dtype=float32, device='native:0')

>>> with xc.device_scope('cuda:0'):
...     b = xc.ones_like(a)
...     b
array([1., 1., 1.], dtype=float32, device='cuda:0')

>>> a + a
array([2., 2., 2.], dtype=float32, device='native:0')
>>> b + b
array([2., 2., 2.], dtype=float32, device='cuda:0')

>>> with xc.device_scope('native'):
...     xc.ones_like(b)
array([1., 1., 1.], dtype=float32, device='native:0')


Using two contexts in one script
--------------------------------

>>> ctx = xc.Context()
>>> with xc.context_scope(ctx):
...     with xc.device_scope('native'):
...         c = xc.ones((3,), xc.float32)
>>> a.device == c.device
False

>>> # This clean up is currently needed...
>>> xc.set_default_device('native')
