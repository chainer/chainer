Tips and FAQs
=============

It takes too long time to compile a computational graph. Can I skip it?
-----------------------------------------------------------------------

Chainer does not compile computational graphs, so you cannot skip it, or, I mean, you have already skipped it :).

It seems you have actually seen on-the-fly compilations of CUDA kernels.
CuPy compiles kernels on demand to make kernels optimized to the number of dimensions and element types of input arguments.
Pre-compilation is not available, because we have to compile an exponential number of kernels to support all CuPy functionalities.
This restriction is unavoidable because Python cannot call CUDA/C++ template functions in generic way.
Note that every framework using CUDA require compilation at some point; the difference between other statically-compiled frameworks (such as cutorch) and Chainer is whether a kernel is compiled at installation or at the first use.

These compilations should run only at the first use of the kernels.
The compiled binaries are cached to the ``$(HOME)/.cupy/kernel_cache`` directory by default.
If you see that compilations run every time you run the same script, then the caching is failed.
Please check that the directory is kept as is between multiple executions of the script.
If your home directory is not suited to caching the kernels (e.g. in case that it uses NFS), change the kernel caching directory by setting the ``CUPY_CACHE_DIR`` environment variable to an appropriate path.
See `CuPy Overview <https://docs-cupy.chainer.org/en/stable/overview.html>`_ for more details.


MNIST example does not converge in CPU mode on Mac OS X
-------------------------------------------------------

.. note::

   Mac OS X is not officially supported.
   Please use it at your own risk.

Many users have reported that MNIST example does not work correctly
when using vecLib as NumPy backend on Mac OS X.
vecLib is the default BLAS library installed on Mac OS X.

We recommend using other BLAS libraries such as `OpenBLAS <http://www.openblas.net/>`_.

To use an alternative BLAS library, it is necessary to reinstall NumPy.
Here is an instruction to install NumPy with OpenBLAS using `Homebrew <https://brew.sh/>`_.

::

   $ brew tap homebrew/science
   $ brew install openblas
   $ brew install numpy --with-openblas

If you want to install NumPy with pip, use `site.cfg <https://github.com/numpy/numpy/blob/master/site.cfg.example>`_ file.

For details of this problem, see `issue #704 <https://github.com/chainer/chainer/issues/704>`_.
