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
See :ref:`Examples <chainerx-install-cuda-support>` below.

.. _chainerx-install-cuda-support:

CUDA support
------------

When installing with the CUDA support, you also need to specify the cuDNN installation path.

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


.. seealso::

    `CuPy installation guide <https://docs-cupy.chainer.org/en/stable/install.html>`_


.. _chainerx-install-examples:

Examples
--------

Install ChainerX without CUDA support:

.. code-block:: console

    $ export CHAINER_BUILD_CHAINERX=1
    $ export MAKEFLAGS=-j8  # Using 8 parallel jobs.
    $ pip install --pre chainer


Install ChainerX depending on CuPy wheel distribution:

.. code-block:: console

    $ pip install --pre cupy_cuda101  # Note: Choose the proper CUDA SDK version number.
    $ export CHAINER_BUILD_CHAINERX=1
    $ export CHAINERX_BUILD_CUDA=1
    $ export CHAINERX_CUDNN_USE_CUPY=1
    $ export MAKEFLAGS=-j8  # Using 8 parallel jobs.
    $ pip install --pre chainer


Install ChainerX with CuPy built from source:

.. code-block:: console

    $ export CHAINER_BUILD_CHAINERX=1
    $ export CHAINERX_BUILD_CUDA=1
    $ export CUDNN_ROOT_DIR=path/to/cudnn
    $ export MAKEFLAGS=-j8  # Using 8 parallel jobs.
    $ pip install --pre cupy
    $ pip install --pre chainer
