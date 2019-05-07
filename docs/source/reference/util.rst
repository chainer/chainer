Utilities
============

.. toctree::
   :maxdepth: 2

Convolution/Deconvolution utilities
------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.utils.get_conv_outsize
   chainer.utils.get_deconv_outsize
   

Utilities across backends
-------------------------

.. module:: chainer.backend
.. currentmodule:: chainer

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.get_device
   chainer.using_device
   chainer.backend.copyto
   chainer.backend.get_array_module

CUDA
----

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
   chainer.backends.cuda.raw
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

iDeep
-----

`iDeep <https://github.com/intel/ideep>`__ is a module that provides NumPy-like API and DNN acceleration using MKL-DNN for Intel CPUs.
See :doc:`../../tips` and :doc:`../../performance` for details.

.. module:: chainer.backends.intel64
.. currentmodule:: chainer

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.backends.intel64.is_ideep_available
   
Common algorithms
-------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.utils.WalkerAlias




Reporter
---------
.. currentmodule:: chainer
.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.Reporter
   chainer.get_current_reporter
   chainer.report
   chainer.report_scope

Summary and DictSummary
------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.Summary
   chainer.DictSummary
   

Sparse utilities
---------------------------

A :class:`chainer.Variable` can be converted into a sparse matrix in e.g.
COO (Coordinate list) format.
A sparse matrix stores the same data as the original object but with a
different internal representation, optimized for efficient operations on
sparse data, i.e. data with many zero elements.

Following are a list of supported sparse matrix formats and utilities for
converting between a :class:`chainer.Variable` and these representations.

.. note::

  Please be aware that only certain functions accept sparse matrices as
  inputs, such as :func:`chainer.functions.sparse_matmul`.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.utils.CooMatrix
   chainer.utils.to_coo
   

Experimental feature annotation
---------------------------------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.utils.experimental

