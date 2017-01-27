CUDA utilities
--------------
.. automodule:: chainer.cuda

Devices
~~~~~~~
.. autofunction:: get_device

CuPy array allocation and copy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: copy
.. autofunction:: to_cpu
.. autofunction:: to_gpu

Kernel definition utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: memoize
.. autofunction:: clear_memo
.. autofunction:: elementwise
.. autofunction:: reduce

CPU/GPU generic code support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: get_array_module

cuDNN support
~~~~~~~~~~~~~
.. autofunction:: set_max_workspace_size
.. autofunction:: get_max_workspace_size
