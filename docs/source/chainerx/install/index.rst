Installation
============

.. _chainerx_install:

ChainerX, or ``chainerx``, can be installed as a top level Python package along with Chainer by configuring the environment variables below.

.. note::

    Chainer must currently be installed from source in order to include ChainerX, but this is expected to change in the near future.

Environment variables
---------------------

Configure the following environment variables before installing Chainer.

========================== ================================================================================================
Environment variable       Description
========================== ================================================================================================
``CHAINER_BUILD_CHAINERX`` ``1`` to build the ``chainerx`` package along with ``chainer``. ``0`` to skip. Default is ``0``.
``CHAINERX_BUILD_CUDA``    ``1`` to build ``chainerx`` with CUDA support. ``0`` to skip. Default is ``0``.
``CUDNN_ROOT_DIR``         Path to your cuDNN installation. Required when ``CHAINERX_BUILD_CUDA=1``.
========================== ================================================================================================

Installing from source
----------------------

Simply run ``pip install --pre chainer`` after configuring the above environment variables.

Example
~~~~~~~

For instance, to install ChainerX without CUDA support, run the following:

.. code-block:: console

    $ export CHAINER_BUILD_CHAINERX=1
    $ export MAKEFLAGS=-j8  # Using 8 parallel jobs.
    $ pip install --pre chainer

CUDA support
------------

When installing with the CUDA support, you also need to specify the cuDNN installation path since CUDA without cuDNN is currently not supported.

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
