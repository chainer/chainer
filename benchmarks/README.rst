Chainer Benchmarks
==================

Benchmarking Chainer with Airspeed Velocity.

Note that CuPy earlier than v3.1.0 or v4.0.0b1 are not supported.

Requirements
------------

* ``asv``
* ``Cython`` (to build CuPy)

Usage
-----

.. code-block:: sh

    # Enable ccache for performance (optional).
    export PATH="/usr/lib/ccache:${PATH}"
    export NVCC="ccache nvcc"

    # Run benchmark against target commit-ish of Chainer and CuPy.
    # Note that specified versions must be a compatible combination.
    # You can use `find_cupy_version.py` helper tool to get appropriate CuPy
    # version for the given Chainer version.
    ./run.sh master master
    ./run.sh v4.0.0b4 v4.0.0b4

    # Compare the benchmark results between two commits to see regression
    # and/or performance improvements in command line.
    alias git_commit='git show --format="%H"'
    asv compare $(git_commit v4.0.0b4) $(git_commit master)

    # Convert the results into HTML.
    # The result will be in `html` directory.
    asv publish

    # Start the HTTP server to browse HTML.
    asv preview
