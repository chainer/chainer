Contribution Guide
==================

This is a guide aimed towards contributors of ChainerX which is mostly implemented in C++.
It describes how to build the project and how to run the test suite so that you can get started contributing.

.. note::

    Please refer to the :ref:`Chainer Contribution Guide <contrib>` for the more general contribution guideline that is not specific to ChainerX.
    E.g. how to download the source code, manage git branches, send pull requests or contribute to Chainer's Python code base.

Building the shared library
---------------------------

You can build the C++ ChainerX project to generate a shared library similar to any other cmake project.
Run the following command from the root of the project to generate ``chainerx_cc/build/chainerx/libchainerx.so``,

.. code-block:: console

    $ cd chainerx_cc
    $ mkdir -p build
    $ cd build
    $ cmake ..
    $ make

The CUDA support is enabled by, either setting ``CHAINERX_BUILD_CUDA=1`` as an environment variable or specifying ``-DCHAINERX_BUILD_CUDA=1`` in ``cmake``.
When building with the CUDA support, either the ``CUDNN_ROOT_DIR`` environment variable or ``-DCUDNN_ROOT_DIR`` is required to locate the cuDNN installation path.

.. note::

    CUDA without cuDNN is currently not supported.

Then, to install the headers and the library, run:

.. code-block:: console

    $ make install

You can specify the installation path using the prefix ``-DCMAKE_INSTALL_PREFIX=<...>`` in ``cmake``.

Running the test suite
----------------------

The test suite can be built by passing ``-DCHAINERX_BUILD_TEST=ON`` to ``cmake``.
It is not built by default.
Once built, run the suite with the following command from within the ``build`` directory.

.. code-block:: console

    $ ctest -V

Coding standards
----------------

The ChainerX C++ coding standard is mostly based on the `Google C++ Style Guide <https://google.github.io/styleguide/cppguide.html>`_ and principles.

Formatting
~~~~~~~~~~

ChainerX is formatted using `clang-format <https://clang.llvm.org/docs/ClangFormat.html>`_.
To fix the formatting in-place, run the following command from the repository root:

.. code-block:: console

    $ scripts/run-clang-format.sh --in-place

Lint checking
~~~~~~~~~~~~~

ChainerX uses the `cpplint <https://github.com/cpplint/cpplint>`_ and `clang-tidy <http://clang.llvm.org/extra/clang-tidy/>`_ for lint checking.
Note that clang-tidy requires that you've finished running ``cmake``.
To run them, run the following commands from the repository root:

.. code-block:: console

    $ scripts/run-cpplint.sh
    $ make clang-tidy

Thread sanitizer
----------------

The thread sanitizer can be used to detect thread-related bugs, such as data races.
To enable the thread sanitizer, pass ``-DCHAINERX_ENABLE_THREAD_SANITIZER=ON`` to ``cmake``.

You can run the test with ``ctest -V`` as usual and you will get warnings if the thread sanitizer detects any issues.

CUDA runtime is known to cause a thread leak error as a false alarm.
In such case, disable the thread leak detection using environment variable ``TSAN_OPTIONS='report_thread_leaks=0'``.

Python contributions and unit tests
-----------------------------------

To test the Python binding, run the following command at the repository root:

.. code-block:: console

    $ pytest

Run tests with coverage:

.. code-block:: console

    $ pytest --cov --no-cov-on-fail --cov-fail-under=80

Run tests without CUDA GPU:

.. code-block:: console

    $ pytest -m 'not cuda'

Test coverage
-------------

We use `gcov <https://gcc.gnu.org/onlinedocs/gcc/Gcov.html>`_ to the measure C++ code coverage.
Build the Python package in ``Debug`` mode, and build C++ test suite as:

.. code-block:: console

    $ python setup.py build --debug --build-temp ./build --build-lib ./build develop
    $ mkdir -p build
    $ cd build
    $ cmake -DCMAKE_BUILD_TYPE=Debug -DCHAINERX_BUILD_PYTHON=1 -DCHAINERX_ENABLE_COVERAGE ..
    $ make

Run both the Python and the C++ test suite:

.. code-block:: console

    $ pytest
    $ cd build
    $ ctest -V

Then find the ``.gcda`` files:

.. code-block:: console

    $ find build -name '*.gcda'

Use the ``gcov`` command to get coverage:

.. code-block:: console

    $ gcov ./build/chainerx/CMakeFiles/chainerx.dir/chainerx.gcda

See generated ``.gcov`` files.

You can also generate HTML coverage reports with `lcov <https://github.com/linux-test-project/lcov>`_. After running tests:

.. code-block:: console

    $ lcov -c -b chainerx -d build/chainerx/ --no-external -o build/coverage.info
    $ genhtml build/coverage.info -o build/coverage

Then open ``build/coverage/index.html`` with any browsers.
