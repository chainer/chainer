# ChainerX

## Installation

For building C++ shared library, run `cmake` with `-DCUDNN_ROOT_DIR`:

```shell-session
$ cmake -DCUDNN_ROOT_DIR=$HOME/.cudnn/active ..
```

To disable CUDA support, either set `CHAINERX_BUILD_CUDA=0` as environment variable or specify `-DCHAINERX_BUILD_CUDA=0` in `cmake`.

## Development

### How to build the C++ shared library

Build `build/chainerx/libchainerx.so` with the following commands.

```shell-session
$ mkdir -p build
$ cd build
$ cmake ..
$ make
```

Again, CUDA support is enabled by default.

To install headers and the library, run:

```shell-session
$ make install
```

To specify the installation path, pass `-DCMAKE_INSTALL_PREFIX=<...>` to `cmake`.

To build the Python binding as a C++ project, pass `-DCHAINERX_BUILD_PYTHON=1` to `cmake`,
then `make` will automatically build the Python binding.

### Python coding style and lint

We use `flake8` and `autopep8`. To install, run:

```
$ pip install hacking flake8 autopep8
```

To format and make changes to Python codes in place, run the following at the repository root:

```
$ autopep8 python tests -r --global-config .pep8 --in-place
```

Lint:

```
$ flake8 python tests
```

### C++ code formatter

We use `clang-format`. To install on Ubuntu, run:

```
$ sudo apt-get install clang-format
```

To format and make changes to C++ codes in place, run the following at the repository root:

```shell-session
$ scripts/run-clang-format.sh --in-place
```

### C++ Lint (cpplint)

We use `cpplint`. To install it, run:

```shell-session
$ pip install cpplint
```

Run the following at the repository root:

```shell-session
$ scripts/run-cpplint.sh
```

### C++ Lint (clang-tidy)

We use `clang-tidy`. To install on Ubuntu, run:

```
$ sudo apt-get install clang clang-tidy
```

Build C++ project beforehand, then run the lint:

```
$ mkdir -p build
$ cd build
$ cmake -DCHAINERX_BUILD_PYTHON=1 ..

$ make clang-tidy
```

### Run the Python test suite

ChainerX requires `chainer` package for Python tests. To install the `chainer` Python package of up-to-date beta version, run the following:

```shell-session
$ pip install chainer --pre
```

To build the `chainerx` Python package in `develop` mode, run the following at the repository root:

```shell-session
$ pip install -e .
```

Run tests with the following command at the repository root:

```shell-session
$ pytest
```

Run tests with coverage:

```shell-session
$ pytest --cov --no-cov-on-fail --cov-fail-under=80
```

Run tests without CUDA GPU:

```shell-session
$ pytest -m 'not cuda'
```

### Run the C++ test suite

The test suite is built by default unless `-DCHAINERX_BUILD_TEST=OFF` is passed to `cmake`.
Run the tests with the following command from within `build`.

```shell-session
$ ctest -V
```

### C++ code coverage

We use `gcov` to measure C++ code coverage.
Build Python package in `Debug` mode, and build C++ test suite as:

```
$ python setup.py build --debug --build-temp ./build --build-lib ./build develop
$ mkdir -p build
$ cd build
$ cmake -DCHAINERX_BUILD_PYTHON=1 -DCHAINERX_ENABLE_COVERAGE ..
$ make
```

Run both Python and C++ test suite:

```shell-session
$ pytest

$ cd build
$ ctest -V
```

then find .gcda files:

```shell-session
$ find build -name '*.gcda'
```

Use `gcov` command to get coverage:

```shell-session
$ gcov ./build/chainerx/CMakeFiles/chainerx.dir/chainerx.gcda
```

See generated .gcov files.

You can also genearte HTML coverage reports with `lcov`. After running tests:

```shell-session
$ lcov -c -b chainerx -d build/chainerx/ --no-external -o build/coverage.info
$ genhtml build/coverage.info -o build/coverage
```

See `build/coverage/index.html` with any browsers.

### Thread sanitizer

Thread sanitizer can be used to detect thread-related bugs, such as data races.
To enable thread sanitizer, pass `-DCHAINERX_ENABLE_THREAD_SANITIZER=ON` to `cmake`.

You can run the test with `ctest -V` as usual and you will get warnings if the thread sanitizer detects any issues.

CUDA runtime is known to cause a thread leak error as a false alarm.
In such case, disable the thread leak detection using environment variable `TSAN_OPTIONS='report_thread_leaks=0'`.
