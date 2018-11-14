Installation
============

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
``CHAINERX_BUILD_CUDA``    ``1`` to build ``chainerx`` with CUDA support. ``0`` to skip. Default is ``1``.
``CUDNN_ROOT_DIR``         Path to your cuDNN installation. Required when ``CHAINERX_BUILD_CUDA=1``.
========================== ================================================================================================

Installing from source
----------------------

Simply run ``pip install .`` from the root directory of the project tree after configuring the above environment variables.

Example
~~~~~~~

For instance, to install ChainerX with CUDA support (using 8 parallel jobs), run the following::

    $ export CHAINER_BUILD_CHAINERX=1
    $ export CUDNN_ROOT_DIR=path/to/cudnn
    $ export MAKEFLAGS=-j8
    $ pip install .

CUDA support
------------

When installing with the CUDA support, you also need to specify the cuDNN installation path as shown in the example above since CUDA without cuDNN is currently not supported.
