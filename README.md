# xChainer

## Build instruction

### Build the core library

Build `build/xchainer/libxchainer.so` with the following commands.

```shell-session
$ ( mkdir -p build && cd build && cmake .. && make )
```

Install headers and the library with `make install`.
To specify the installation path, pass `-DCMAKE_INSTALL_PREFIX=<...>` to `cmake`.

### Build the Python binding

To install the `xchainer` Python package in `Release` mode, run the following at the repository root:

```shell-session
$ pip install .
```

To build the Python binding as a C++ project, pass `-DBUILD_PYTHON=1` to `cmake`,
then `make` will automatically build the Python binding.

## Information for developers

### Python format and lint

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

### C++ format

We use `clang-format`. To install on Ubuntu, run:

```
$ sudo apt-get install clang-format
```

To format and make changes to C++ codes in place, run the following at the repository root:

zsh)

```shell-session
$ clang-format -i xchainer/**/*.{cc,h,cu}
```

bash)

```shell-session
$ find xchainer \( -name '*.cc' -o -name '*.h' -o -name '*.cu' \) -type f -print0 | xargs -0 clang-format -i
```

### C++ Lint

We use `clang-tidy`. To install on Ubuntu, run:

```
$ sudo apt-get install clang clang-tidy
```

Build C++ project beforehand, then run the lint:

```
$ ( mkdir -p build && cd build && cmake -DBUILD_PYTHON=1 .. && make clang-tidy )
```

### Run the Python test suite

To build the `xchainer` Python package in `develop` mode, run the following at the repository root:

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

### Run the C++ test suite

The test suite is built by default unless `-DENABLE_TEST=OFF` is passed to `cmake`.
Run the tests with the following command from within `build`.

```shell-session
$ ctest -V
```

### C++ code coverage

We use `gcov` to measure C++ code coverage.
Build Python package in `Debug` mode, and build C++ test suite as:

```
$ python setup.py build --debug --build-temp ./build --build-lib ./build develop
$ ( mkdir -p build && cd build && cmake -DBUILD_PYTHON=1 -DENABLE_COVERAGE .. && make )
```

Run both Python and C++ test suite:

```shell-session
$ pytest && ( cd build && ctest -V )
```

then find .gcda files:

```shell-session
find build -name '*.gcda'
```

Use `gcov` command to get coverage:

```shell-session
gcov ./build/xchainer/CMakeFiles/xchainer.dir/xchainer.gcda
```

See generated .gcov files.

You can also genearte HTML coverage reports with `lcov`. After running tests:

```shell-session
lcov -c -b xchainer -d build/xchainer/ --no-external -o build/coverage.info
genhtml build/coverage.info -o build/coverage
```

See `build/coverage/index.html` with any browsers.
