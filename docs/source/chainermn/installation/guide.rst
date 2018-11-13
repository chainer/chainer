.. _chainermn_installation:

Installation Guide
==================

Requirements
------------

ChainerMN depends on the following software libraries:
CUDA-Aware MPI, NVIDIA NCCL, and a few Python packages including CuPy and MPI4py.

.. note::

    In Chainer v5, ChainerMN became a part of Chainer package.
    Installing Chainer (``pip install chainer``) automatically makes ChainerMN available.
    Note that you still need to separately install requirements described below to actually run code using ChainerMN.

    Before upgrading from Chainer v4 to v5 or later, make sure to remove existing ``chainermn`` package (``pip uninstall chainermn``).

.. _mpi-install:

CUDA-Aware MPI
~~~~~~~~~~~~~~

ChainerMN relies on MPI.
In particular, for efficient communication between GPUs, it uses CUDA-aware MPI.
For details about CUDA-aware MPI, see `this introduction article <https://devblogs.nvidia.com/parallelforall/introduction-cuda-aware-mpi/>`_.
(If you use only the CPU mode, MPI does not need to be CUDA-Aware. See :ref:`non-gpu-env` for more details.)

The CUDA-aware features depend on several MPI packages, which need to be configured and built properly.
The following are examples of Open MPI and MVAPICH.

Open MPI (for details, see `Open MPI's official instructions <https://www.open-mpi.org/faq/?category=building#build-cuda>`__)::

  $ ./configure --with-cuda
  $ make -j4
  $ sudo make install

MVAPICH (for details, see `Mvapich's official instructions <http://mvapich.cse.ohio-state.edu/static/media/mvapich/mvapich2-2.0-userguide.html#x1-120004.5>`__)::

  $ ./configure --enable-cuda
  $ make -j4
  $ sudo make install
  $ export MV2_USE_CUDA=1  # Should be set all the time when using ChainerMN

.. _nccl-install:

NCCL
~~~~

.. note::

    If you are installing CuPy using wheels (i.e., ``pip install cupy-cudaXX`` where ``XX`` is the CUDA version), you don't have to install NCCL manually.
    The latest NCCL 2.x library is bundled with CuPy wheels.

    See `CuPy Installation Guide <https://docs-cupy.chainer.org/en/stable/install.html>`__ for the detailed steps to install CuPy.

To enable efficient intra- and inter-node GPU-to-GPU communication,
we use `NVIDIA Collective Communications Library (NCCL) <https://developer.nvidia.com/nccl>`_.
See `NCCL's official instructions <http://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html#downloadnccl>`__ for installation.

ChainerMN requires NCCL even if you have only one GPU per node. The
only exception is when you run ChainerMN on CPU-only environments. See
:ref:`non-gpu-env` for more details.

.. note::

   We reccomend NCCL 2 but NCCL 1 can be used.
   However, for NCCL 1, ``PureNcclCommunicator`` is not supported in ChainerMN.
   If you use NCCL 1, please properly configure environment variables to expose NCCL both when you install and use ChainerMN.
   Typical configurations should look like the following::

     export NCCL_ROOT=<path to NCCL directory>
     export CPATH=$NCCL_ROOT/include:$CPATH
     export LD_LIBRARY_PATH=$NCCL_ROOT/lib/:$LD_LIBRARY_PATH
     export LIBRARY_PATH=$NCCL_ROOT/lib/:$LIBRARY_PATH

   If you change the version of NCCL installed, you have to reinstall CuPy. Because, current ChainerMN applies CuPy to use NCCL.
   See `CuPy official instructions <https://docs-cupy.chainer.org/en/stable/install.html#reinstall-cupy>`__ for reinstalltion.

.. _mpi4py-install:


MPI4py
~~~~~~

You can install MPI4py by::

  $ pip install mpi4py

Please make be sure to properly configure environment variables so that MPI is available at installation time, because MPI4py links to MPI library at installation time.
In particular, if you have multiple MPI implementations installed in your environment, please expose the implementation that you want to use both when you install and use ChainerMN.

.. _cupy-install:

CuPy
~~~~

Chainer and ChainerMN rely on CuPy to use GPUs.
Please refer to `CuPy Installation Guide <https://docs-cupy.chainer.org/en/stable/install.html>`__ for the detailed steps to install CuPy.

In most cases it is recommended to install CuPy using wheel distribution (precompiled binary) rather than source distribution.
If you are installing from source, NCCL library must be installed before installing CuPy to enable NCCL feature in CuPy.
Refer to :ref:`nccl-install` for the installation steps of NCCL library.
See :ref:`check-nccl`, if you want to check whether NCCL is enabled in your CuPy.

Chainer and ChainerMN can be installed without CuPy, in which case the corresponding features are not available.
See :ref:`non-gpu-env` for more details.


Tested Environments
-------------------

We tested ChainerMN on all the following environments.

* OS

  * Ubuntu 14.04 LTS 64bit
  * Ubuntu 16.04 LTS 64bit

* Python 2.7.13, 3.5.1, 3.6.1
* MPI

  * openmpi 1.10.7, 2.1.2

* MPI4py 3.0.0
* NCCL 2.2.13

.. _non-gpu-env:

Installation on Non-GPU Environments
------------------------------------

Users who want to try ChainerMN in CPU-only environment may skip installation of CuPy.
Non-GPU set up may not be performant as GPU-enabled set up,
but would be useful for testing or debugging training program
in non-GPU environment such as laptops or CI jobs.

In this case, the MPI does not have to be CUDA-aware.
Only ``naive`` communicator works with the CPU mode.
