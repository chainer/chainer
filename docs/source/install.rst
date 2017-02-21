Install Guide
=============

.. _before_install:

Before installing Chainer
-------------------------

We recommend these platforms.

* `Ubuntu <http://www.ubuntu.com/>`_ 14.04 LTS 64bit
* `CentOS <https://www.centos.org/>`_ 7 64bit

Chainer is supported on Python 2.7.6+, 3.4.3+, 3.5.1+.
Chainer uses C++ compiler such as g++.
You need to install it before installing Chainer.
This is typical installation method for each platform::

  # Ubuntu 14.04
  $ apt-get install g++

  # CentOS 7
  $ yum install gcc-c++

If you use old ``setuptools``, upgrade it::

  $ pip install -U setuptools


Install Chainer
---------------

Chainer depends on these Python packages:

* `NumPy <http://www.numpy.org/>`_ 1.9, 1.10, 1.11
* `Six <https://pythonhosted.org/six/>`_ 1.9

CUDA support

* `CUDA <https://developer.nvidia.com/cuda-zone>`_ 6.5, 7.0, 7.5, 8.0
* `filelock <https://filelock.readthedocs.org>`_

cuDNN support

* `cuDNN <https://developer.nvidia.com/cudnn>`_ v2, v3, v4, v5, v5.1

Caffe model support

* `Protocol Buffers <https://developers.google.com/protocol-buffers/>`_
* protobuf>=3.0.0 is required for Py3

All these libraries are automatically installed with ``pip`` or ``setup.py``.

Image dataset is optional

* `Pillow <https://pillow.readthedocs.io/>`_

HDF5 serialization is optional

* `h5py <http://www.h5py.org/>`_ 2.5.0


Install Chainer via pip
~~~~~~~~~~~~~~~~~~~~~~~

We recommend to install Chainer via pip::

  $ pip install chainer


Install Chainer from source
~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use ``setup.py`` to install Chainer from source::

  $ tar zxf chainer-x.x.x.tar.gz
  $ cd chainer-x.x.x
  $ python setup.py install


.. _install_error:

When an error occurs...
~~~~~~~~~~~~~~~~~~~~~~~

Use ``-vvvv`` option with ``pip`` command.
That shows all logs of installation. It may helps you::

  $ pip install chainer -vvvv


.. _install_cuda:

Enable CUDA/cuDNN support
~~~~~~~~~~~~~~~~~~~~~~~~~

In order to enable CUDA support, you first have to install `CuPy <https://docs.cupy-ndarray.org/>`_.
It requires CUDA Toolkit and ``CUDA_PATH`` environment variable.
If CuPy is correctly installed, Chainer will automatically enable CUDA support::

  $ pip install cupy

In order to enable cuDNN support as well, you have to install cuDNN before installing CuPy.
If CuPy is correctly set up with cuDNN, Chainer will automatically enable cuDNN support.

.. note::

   CuPy installer looks up ``CUDA_PATH`` environment variable first.
   If it is empty, the installer looks for ``nvcc`` command from ``PATH`` environment variable and use its parent directory as the root directory of CUDA installation.
   If ``nvcc`` command is also not found, the installer tries to use the default directory for Ubuntu ``/usr/local/cuda``.

If you installed CUDA into a non-default directory, you need to specify the directory with ``CUDA_PATH`` environment variable::

  $ CUDA_PATH=/opt/nvidia/cuda pip install cupy


.. warning::

   If you want to use ``sudo`` to install CuPy, note that ``sudo`` command initializes all environment variables.
   Please specify ``CUDA_PATH`` environment variable inside ``sudo`` like this::

      $ sudo CUDA_PATH=/opt/nvidia/cuda pip install cupy


Support image dataset
~~~~~~~~~~~~~~~~~~~~~

Install Pillow manually to activate image dataset.
This feature is optional::

  $ pip install pillow


Support HDF5 serialization
~~~~~~~~~~~~~~~~~~~~~~~~~~

Install h5py manually to activate HDF5 serialization.
This feature is optional::

  $ pip install h5py

Before installing h5py, you need to install libhdf5.
It depends on your environment::

  # Ubuntu 14.04
  $ apt-get install libhdf5-dev

  # CentOS 7
  $ yum -y install epel-release
  $ yum install hdf5-devel


Uninstall Chainer
-----------------

Use pip to uninstall Chainer::

  $ pip uninstall chainer

.. note::

   When you upgrade Chainer, ``pip`` sometimes installed various version of Chainer in ``site-packages``.
   Please uninstall it repeatedly until ``pip`` returns an error.


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

We provide the official Docker image.
Use `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`_ command to run Chainer image with GPU.
You can login to the environment with bash, and run the Python interpreter::

  $ nvidia-docker run -it chainer/chainer /bin/bash

Or, run the interpreter directly::

  $ nvidia-docker run -it chainer/chainer /usr/bin/python


What "recommend" means?
-----------------------

We tests Chainer automatically with Jenkins.
All supported environments are tested in this environment.
We cannot guarantee that Chainer works on other environments.


FAQ
---

The installer says "hdf5.h is not found"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You don't have libhdf5.
Please install hdf5.
See :ref:`before_install`.


MemoryError happens
~~~~~~~~~~~~~~~~~~~

You maybe failed to install Cython.
Please install it manually.
See :ref:`install_error`.


Examples says "cuDNN is not enabled"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You failed to build CuPy with cuDNN.
If you don't need cuDNN, ignore this message.
Otherwise, retry to install CuPy with cuDNN.
``-vvvv`` option helps you.
There is no need of re-installing Chainer itself.
See :ref:`install_cudnn`.
