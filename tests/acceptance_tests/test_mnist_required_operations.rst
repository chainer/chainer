Acceptance tests for MNIST required operations
==============================================

>>> import xchainer as xc
>>> import numpy as np

Dot
---

>>> a = xc.Array((2, 3), xc.float32, [1, 2, 3, 4, 5, 6])
>>> b = xc.Array((3,), xc.float32, [4, 2, 5])

>>> a == b
array([[False,  True, False],
       [ True, False, False]], shape=(2, 3), dtype=bool, device='native:0')
>>> xc.equal(a, b)
array([[False,  True, False],
       [ True, False, False]], shape=(2, 3), dtype=bool, device='native:0')

Subtract
--------

Divide
------

Max
---

Argmax
------

Log
---

Exp
---

Negative
--------

Log of Softmax
--------------

AsType
------

Acceptance tests for Refactoring
================================

Arange
------

Squeeze
-------

xchainer.array
--------------

xchainer.dtype accepts string
-----------------------------

