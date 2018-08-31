Acceptance tests for pluggable backend system on GPU
====================================================

Using two backends in one script
--------------------------------

>>> import chainerx as xc

>>> a = xc.ones((3,), xc.float32)
>>> a
array([1., 1., 1.], shape=(3,), dtype=float32, device='native:0')

>>> with xc.device_scope('cuda:0'):
...     b = xc.ones_like(a)
...     b
array([1., 1., 1.], shape=(3,), dtype=float32, device='cuda:0')

>>> a + a
array([2., 2., 2.], shape=(3,), dtype=float32, device='native:0')
>>> b + b
array([2., 2., 2.], shape=(3,), dtype=float32, device='cuda:0')

>>> with xc.device_scope('native'):
...     xc.ones_like(b)
array([1., 1., 1.], shape=(3,), dtype=float32, device='native:0')

Transfer array onto a different device
--------------------------------------

>>> with xc.device_scope('native'):  # Allocate arrays on CPU
...     a = xc.ndarray((2, 3), xc.float32, [0, 1, 2, 3, 4, 5]).require_grad()
>>> a.device
native:0
>>> ag = a.to_device('cuda')  # Transfer onto CUDA device
>>> ag.device
cuda:0
>>> ag.is_backprop_required()
True

Backward on a graph with multiple devices
-----------------------------------------

>>> with xc.device_scope('native'):  # Allocate arrays on CPU
...     a = xc.ndarray((2, 3), xc.float32, [0, 1, 2, 3, 4, 5]).require_grad()
...     b = xc.ndarray((2, 3), xc.float32, [0, 1, 2, 3, 4, 5]).require_grad()
...
>>> ag = a.to_device('cuda')  # Transfer onto CUDA device
>>> y = ag * b  # Arithmetics between different devices is not allowed
Traceback (most recent call last):
...
chainerx.DeviceError: Device (cuda:0) is not compatible with array's device (native:0).
>>> bg = b.to_device('cuda')
>>> y = ag * bg
>>> _ = y.require_grad()
>>> y.device
cuda:0
>>> xc.backward(y)
>>> y.get_grad().device
cuda:0
>>> a.get_grad().device
native:0
