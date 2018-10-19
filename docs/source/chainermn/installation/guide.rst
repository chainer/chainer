.. _chainermn_installation:

Installation Guide
==================

Requirements
------------
In addition to Chainer, ChainerMN depends on the following software libraries:
CUDA-Aware MPI, NVIDIA NCCL, and a few Python packages including CuPy and MPI4py.


Chainer
~~~~~~~

ChainerMN adds distributed training features to Chainer;
thus it naturally requires Chainer.
Please refer to `Chainer's official instructions <http://docs.chainer.org/en/latest/install.html>`__ to install.

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

To enable efficient intra- and inter-node GPU-to-GPU communication,
we use `NVIDIA Collective Communications Library (NCCL) <https://developer.nvidia.com/nccl>`_.
See `NCCL's official instructions <http://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html#downloadnccl>`__ for installation.

ChainerMN requires NCCL even if you have only one GPU per node. The
only exception is when you run ChainerMN on CPU-only environments. See
:ref:`non-gpu-env` for more details.

.. note::

   We reccomend NCCL 2 but NCCL 1 can be used.
   When you use CUDA 7.0 and 7.5, please install NCCL 1 because NCCL 2 is not supported with CUDA 7.0 and 7.5.
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

ChainerMN depends on a few Python packages, which are
automatically installed when you install ChainerMN.

However, among them, we need to be a little careful about MPI4py.
It links to MPI at installation time, so please be sure
to properly configure environment variables
so that MPI is available at installation time.
In particular, if you have multiple MPI implementations in your environment,
please expose the implementation that you want to use
both when you install and use ChainerMN.

.. _cupy-install:

CuPy
~~~~

Chainer and ChainerMN rely on CuPy to use GPUs. 
Please refer to `Chainer's official instructions <https://docs-cupy.chainer.org/en/stable/install.html>`__ to install.
CuPy requires NCCL to be enabled.
See :ref:`check-nccl`, if you want to check whether NCCL is enabled in CuPy.

Chainer and ChainerMN can be installed without CuPy, in which case the corresponding features are not available. 
See :ref:`non-gpu-env` for more details.


Tested Environments
~~~~~~~~~~~~~~~~~~~

We tested ChainerMN on all the following environments.

* OS
  
  * Ubuntu 14.04 LTS 64bit
  * Ubuntu 16.04 LTS 64bit

* Python 2.7.13 3.5.1 3.6.1
* Chainer 3.5.0 4.4.0
* CuPy 2.5.0 4.4.0
* MPI

  * openmpi 1.10.7 2.1.2

* MPI4py 3.0.0
* NCCL 2.2.13
  
.. _chainermn-install:

Installation
------------

Install via pip
~~~~~~~~~~~~~~~

We recommend to install ChainerMN via :command:`pip`::

  $ pip install chainermn

NOTE: If you need :command:`sudo` to use pip, you should be careful
about environment variables.  The :command:`sudo` command DOES NOT
inherit the environment, and thus you need to specify the variables
explicitly. ::

  $ sudo CPATH=${CPATH} LIBRARY_PATH=${LIBRARY_PATH} pip install chainermn


.. _install-from-source:
  
Install from Source
~~~~~~~~~~~~~~~~~~~

You can use ``setup.py`` to install ChainerMN from source::

  $ tar zxf chainermn-x.y.z.tar.gz
  $ cd chainermn-x.y.z
  $ python setup.py install

.. _non-gpu-env:
  
Non-GPU environments
~~~~~~~~~~~~~~~~~~~~

Users who want to try ChainerMN in CPU-only environment may skip installation of CuPy.
Non-GPU set up may not be performant as GPU-enabled set up,
but would be useful for testing or debugging training program
in non-GPU environment such as laptops or CI jobs.

In this case, the MPI does not have to be CUDA-aware.
Only ``naive`` communicator works with the CPU mode.

.. note::

   Current version of ChainerMN does not need ``--no-nccl`` flag 
   for CPU-only environment at installation any more. 
   It would be just ignored.
