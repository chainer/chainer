CUDA utilities
--------------
.. automodule:: chainer.backends.cuda

.. currentmodule:: /

Devices
~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.backends.cuda.get_device
   chainer.backends.cuda.get_device_from_id
   chainer.backends.cuda.get_device_from_array

CuPy array allocation and copy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.backends.cuda.copy
   chainer.backends.cuda.to_cpu
   chainer.backends.cuda.to_gpu

Kernel definition utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.backends.cuda.memoize
   chainer.backends.cuda.clear_memo
   chainer.backends.cuda.elementwise
   chainer.backends.cuda.reduce

CPU/GPU generic code support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.backends.cuda.get_array_module

cuDNN support
~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.backends.cuda.set_max_workspace_size
   chainer.backends.cuda.get_max_workspace_size
