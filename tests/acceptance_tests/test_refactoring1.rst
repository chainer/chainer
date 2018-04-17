Acceptance tests for Refactoring1
=================================

>>> import xchainer as xc
>>> import numpy as np

Arange
------

>>> xc.arange(3)
array([0, 1, 2], shape=(3,), dtype=int64, device='native:0')
>>> xc.arange(start=0.2, stop=3.2)
array([0.2, 1.2, 2.2], shape=(3,), dtype=float64, device='native:0')
>>> xc.arange(start=0.2, stop=1.0, step=0.2)
array([0.2, 0.4, 0.6, 0.8], shape=(4,), dtype=float64, device='native:0')
>>> xc.arange(start=0, stop=1.0, step=0.25, dtype=xc.float32)
array([0.  , 0.25, 0.5 , 0.75], shape=(4,), dtype=float32, device='native:0')

>>> xc.arange(3, device='native:1')
array([0, 1, 2], shape=(3,), dtype=int64, device='native:1')

Squeeze
-------

>>> a = xc.arange(3).reshape(1, 3, 1, 1)
>>> xc.squeeze(a)
array([0, 1, 2], shape=(3,), dtype=int64, device='native:0')
>>> xc.squeeze(a, axis=(2, 3))
array([[0, 1, 2]], shape=(1, 3), dtype=int64, device='native:0')

xchainer.array
--------------

.. TODO(sonots): Fix to take care of python list data types

>>> xc.array([0, 1, 2])
array([0., 1., 2.], shape=(3,), dtype=float64, device='native:0')

>>> xc.array(np.arange(3))
array([0, 1, 2], shape=(3,), dtype=int64, device='native:0')

>>> xc.array(xc.arange(3))
array([0, 1, 2], shape=(3,), dtype=int64, device='native:0')
