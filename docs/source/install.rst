.. _install-guide:

Installation Guide
==================

Recommended Environments
------------------------

We recommend these Linux distributions.

* `Ubuntu <https://www.ubuntu.com/>`__ 14.04/16.04 LTS 64bit
* `CentOS <https://www.centos.org/>`__ 7 64bit

The following versions of Python can be used: 2.7.6+, 3.4.3+, 3.5.1+, and 3.6.0+.

.. note::

   We are testing Chainer automatically with Jenkins, where all the above *recommended* environments are tested.
   We cannot guarantee that Chainer works on other environments including Windows and macOS (especially with CUDA support), even if Chainer looks running correctly.



Dependencies
------------

Before installing Chainer, we recommend to upgrade ``setuptools`` if you are using an old one::

  $ pip install -U setuptools

The following Python packages are required to install Chainer.
The latest version of each package will automatically be installed if missing.

* `NumPy <http://www.numpy.org/>`__ 1.9, 1.10, 1.11, 1.12, 1.13
* `Six <https://pythonhosted.org/six/>`__ 1.9+

The following packages are optional dependencies.
Chainer can be installed without them, in which case the corresponding features are not available.

* CUDA/cuDNN support

  * `cupy <https://cupy.chainer.org/>`__ 4.0+

* Caffe model support

  * `protobuf <https://developers.google.com/protocol-buffers/>`__ 3.0+

* Image dataset support

  * `pillow <https://pillow.readthedocs.io/>`__ 2.3+

* HDF5 serialization support

  * `h5py <http://www.h5py.org/>`__ 2.5+


Install Chainer
---------------

Install Chainer via pip
~~~~~~~~~~~~~~~~~~~~~~~

We recommend to install Chainer via pip::

  $ pip install chainer

.. note::

   Any optional dependencies (including CuPy) can be added after installing Chainer.
   Chainer automatically detects the available packages and enables/disables the optional features appropriately.


Install Chainer from source
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tarball of the source tree is available via ``pip download chainer`` or from `the release notes page <https://github.com/chainer/chainer/releases>`_.
You can use ``setup.py`` to install Chainer from the tarball::

  $ tar zxf chainer-x.x.x.tar.gz
  $ cd chainer-x.x.x
  $ python setup.py install

You can also install the development version of Chainer from a cloned Git repository::

  $ git clone https://github.com/chainer/chainer.git
  $ cd chainer
  $ python setup.py install


.. _install_error:

When an error occurs...
~~~~~~~~~~~~~~~~~~~~~~~

Use ``-vvvv`` option with ``pip`` command.
That shows all logs of installation.
It may help you::

  $ pip install chainer -vvvv


.. _install_cuda:

Enable CUDA/cuDNN support
~~~~~~~~~~~~~~~~~~~~~~~~~

In order to enable CUDA support, you have to install `CuPy <https://cupy.chainer.org/>`_ manually.
If you also want to use cuDNN, you have to install CuPy with cuDNN support.
See `CuPy's installation guide <https://docs-cupy.chainer.org/en/latest/install.html>`_ to install CuPy.
Once CuPy is correctly set up, Chainer will automatically enable CUDA support.

You can refer to the following flags to confirm if CUDA/cuDNN support is actually available.

``chainer.cuda.available``
   ``True`` if Chainer successfully imports :mod:`cupy`.
``chainer.cuda.cudnn_enabled``
   ``True`` if cuDNN support is available.


Support image dataset
~~~~~~~~~~~~~~~~~~~~~

Install Pillow manually to activate image dataset feature::

  $ pip install pillow

Note that this feature is optional.

.. _hdf5-support:

Support HDF5 serialization
~~~~~~~~~~~~~~~~~~~~~~~~~~

Install h5py manually to activate HDF5 serialization::

  $ pip install h5py

Before installing h5py, you need to install libhdf5.
The way to install it depends on your environment::

  # Ubuntu 14.04/16.04
  $ apt-get install libhdf5-dev

  # CentOS 7
  $ yum -y install epel-release
  $ yum install hdf5-devel

Note that this feature is optional.


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

The installer says "hdf5.h is not found"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You don't have libhdf5.
Please install it first.
See :ref:`hdf5-support`.


Examples say "cuDNN is not enabled"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You failed to build CuPy with cuDNN.
If you don't need cuDNN, ignore this message.
Otherwise, retry to install CuPy with cuDNN.
``-vvvv`` option helps you.
There is no need of re-installing Chainer itself.
See `CuPy's installation guide <https://docs-cupy.chainer.org/en/latest/install.html>`_ for more details.
