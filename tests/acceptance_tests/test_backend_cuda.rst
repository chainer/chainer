Acceptance tests for pluggable backend system on GPU
====================================================

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
