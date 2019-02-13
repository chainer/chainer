.. _install-guide:

Installation
============

Recommended Environments
------------------------

We recommend the following Linux distributions.

* `Ubuntu <https://www.ubuntu.com/>`_ 14.04 / 16.04 LTS (64-bit)
* `CentOS <https://www.centos.org/>`_ 7 (64-bit)

.. note::

   We are automatically testing Chainer on all the recommended environments above.
   We cannot guarantee that Chainer works on other environments including Windows and macOS (especially with CUDA support), even if Chainer may seem to be running correctly.


Requirements
------------

You need to have the following components to use Chainer.

* `Python <https://python.org/>`_
    * Supported Versions: 2.7.6+, 3.4.3+, 3.5.1+, 3.6.0+ and 3.7.0+.
* `NumPy <http://www.numpy.org/>`_
    * Supported Versions: 1.9, 1.10, 1.11, 1.12, 1.13, 1.14, 1.15 and 1.16.
    * NumPy will be installed automatically during the installation of Chainer.

Before installing Chainer, we recommend you to upgrade ``setuptools`` and ``pip``::

  $ pip install -U setuptools pip

Hardware Acceleration Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can accelerate performance of Chainer by installing the following optional components.

* NVIDIA CUDA / cuDNN
    * `CuPy <https://cupy.chainer.org/>`_ 5.0+
    * See `CuPy Installation Guide <https://docs-cupy.chainer.org/en/latest/install.html>`__ for instructions.

* Intel CPU (experimental)
    * `iDeep <https://github.com/intel/ideep>`_ 2.0.0.post3+
    * See :doc:`tips` for instructions.

Optional Features
~~~~~~~~~~~~~~~~~

The following packages are optional dependencies.
Chainer can be installed without them, in which case the corresponding features are not available.

* Image dataset support
    * `pillow <https://pillow.readthedocs.io/>`__ 2.3+
    * Run ``pip install pillow`` to install.
* HDF5 serialization support
    * `h5py <http://www.h5py.org/>`__ 2.5+
    * Run ``pip install h5py`` to install.
* Distributed Deep Learning using ChainerMN
    * CUDA-aware MPI
    * `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`__
    * See :ref:`ChainerMN installation guide <chainermn_installation>` for installation instructions.


Install Chainer
---------------

Using pip
~~~~~~~~~

We recommend to install Chainer via pip::

  $ pip install chainer

.. note::

   Any optional dependencies (including CuPy) can be added after installing Chainer.
   Chainer automatically detects the available packages and enables/disables the optional features appropriately.

Using Tarball
~~~~~~~~~~~~~

The tarball of the source tree is available via ``pip download chainer`` or from `the release notes page <https://github.com/chainer/chainer/releases>`_.
You can install Chainer from the tarball::

  $ pip install chainer-x.x.x.tar.gz

You can also install the development version of Chainer from a cloned Git repository::

  $ git clone https://github.com/chainer/chainer.git
  $ cd chainer
  $ pip install .

Enable CUDA/cuDNN support
~~~~~~~~~~~~~~~~~~~~~~~~~

In order to enable CUDA support, you have to install `CuPy <https://cupy.chainer.org/>`_ manually.
If you also want to use cuDNN, you have to install CuPy with cuDNN support.
See `CuPy's installation guide <https://docs-cupy.chainer.org/en/latest/install.html>`__ to install CuPy.
Once CuPy is correctly set up, Chainer will automatically enable CUDA support.

You can refer to the following flags to confirm if CUDA/cuDNN support is actually available.

``chainer.backends.cuda.available``
   ``True`` if Chainer successfully imports :mod:`cupy`.
``chainer.backends.cuda.cudnn_enabled``
   ``True`` if cuDNN support is available.


Google Colaboratory
~~~~~~~~~~~~~~~~~~~

You can install Chainer and CuPy using the following snippet on `Google Colaboratory <https://colab.research.google.com/>`_::

   !curl https://colab.chainer.org/install | sh -

See `chainer/google-colaboratory <https://github.com/chainer/google-colaboratory>`_ for more details and examples.

Uninstall Chainer
-----------------

Use pip to uninstall Chainer::

  $ pip uninstall chainer

.. note::

   When you upgrade Chainer, ``pip`` sometimes install the new version without removing the old one in ``site-packages``.
   In this case, ``pip uninstall`` only removes the latest one.
   To ensure that Chainer is completely removed, run the above command repeatedly until ``pip`` returns an error.


Upgrade Chainer
---------------

Just use ``pip`` with ``-U`` option::

  $ pip install -U chainer


Reinstall Chainer
-----------------

If you want to reinstall Chainer, please uninstall Chainer and then install it.
We recommend to use ``--no-cache-dir`` option as ``pip`` sometimes uses cache::

  $ pip uninstall chainer
  $ pip install chainer --no-cache-dir


Run Chainer with Docker
-----------------------

We are providing the official Docker image.
Use `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`_ command to run Chainer image with GPU.
You can login to the environment with bash, and run the Python interpreter::

  $ nvidia-docker run -it chainer/chainer /bin/bash

Or run the interpreter directly::

  $ nvidia-docker run -it chainer/chainer /usr/bin/python


FAQ
---

Warning message "cuDNN is not enabled" appears
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You failed to build CuPy with cuDNN.
If you don't need cuDNN, ignore this message.
Otherwise, retry to install CuPy with cuDNN.
``pip install -vvvv`` option helps you.
There is no need of re-installing Chainer itself.
See `CuPy's installation guide <https://docs-cupy.chainer.org/en/latest/install.html>`__ for more details.

CuPy always raises ``cupy.cuda.compiler.CompileException``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See FAQ section of `CuPy's installation guide <https://docs-cupy.chainer.org/en/latest/install.html>`__ for details.

h5py installation failed
~~~~~~~~~~~~~~~~~~~~~~~~

If the installation failed with error saying ``hdf5.h is not found``, you need to install ``libhdf5`` first.
The way to install it depends on your environment::

  # Ubuntu 14.04/16.04
  $ apt-get install libhdf5-dev

  # CentOS 7
  $ yum -y install epel-release
  $ yum install hdf5-devel

Note that ``h5py`` is not required unless you need HDF5 serialization support.
