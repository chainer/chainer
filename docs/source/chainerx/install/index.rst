Installation
============

.. _chainerx_install:

ChainerX, or ``chainerx``, can be installed as a top level Python package along with Chainer by configuring the environment variables below.

.. note::

    Chainer must currently be installed from source in order to include ChainerX, but this is expected to change in the near future.

Installing from source
----------------------

The following environment variables are available for building ChainerX from source.


=========================== ========================================================================================================
Environment variable        Description
=========================== ========================================================================================================
``CHAINER_BUILD_CHAINERX``  ``1`` to build the ``chainerx`` package along with ``chainer``. ``0`` to skip. Default is ``0``.
``CHAINERX_BUILD_CUDA``     ``1`` to build ``chainerx`` with CUDA support. ``0`` to skip. Default is ``0``.
                            See also :ref:`CUDA support <chainerx-install-cuda-support>` section below.
``CHAINERX_ENABLE_BLAS``    ``1`` to make BLAS enabled. ``0`` to disabled. Default is ``1``.
=========================== ========================================================================================================


Simply run ``pip install --pre chainer`` after configuring the above environment variables.

For instance, to install ChainerX without CUDA support, run the following:

.. code-block:: console

    $ export CHAINER_BUILD_CHAINERX=1
    $ export MAKEFLAGS=-j8  # Using 8 parallel jobs.
    $ pip install --pre chainer


.. _chainerx-install-cuda-support:

CUDA support
------------

When installing with the CUDA support, you also need to specify the cuDNN installation path since CUDA without cuDNN is currently not supported.

You can specify either of the following environment variables to specify where to look for cuDNN installation.

=========================== ========================================================================================================
Environment variable        Description
=========================== ========================================================================================================
``CUDNN_ROOT_DIR``          Path to your cuDNN installation.
``CHAINERX_CUDNN_USE_CUPY`` ``1`` to search for cuDNN library and include files in existing `CuPy <https://docs-cupy.chainer.org/>`_
                            installation.
                            Only applicable for CuPy installed via wheel (binary) distribution.
                            Other variables related to cuDNN paths (such as ``CUDNN_ROOT_DIR``) are ignored.
                            Be warned that the resulting executable will be invalidated if CuPy is uninstalled, moved or
                            replaced.
=========================== ========================================================================================================

To support the :ref:`NumPy/CuPy fallback <chainerx-tutorial-numpy-cupy-fallback>` mechanism, currently ChainerX with the CUDA support requires CuPy to be installed together.

.. note::

    For ChainerX, we suggest that you do not install CuPy with a CuPy wheel (precompiled binary) package because it contains a cuDNN library.
    Installation would fail if the versions of the cuDNN library contained in the CuPy wheel package and the one specified in `CUDNN_ROOT_DIR` were different.

.. code-block:: console

    $ export CHAINER_BUILD_CHAINERX=1
    $ export CHAINERX_BUILD_CUDA=1
    $ export CUDNN_ROOT_DIR=path/to/cudnn
    $ export MAKEFLAGS=-j8  # Using 8 parallel jobs.
    $ pip install --pre cupy
    $ pip install --pre chainer
