# xChainer

## Build instruction

### Build the core library

Build `build/xchainer/libxchainer.so` with the following commands.

```shell-session
$ mkdir build
$ cd build
$ cmake ..
$ make
```

Install headers and the library with `make install`.
To specify the installation path, pass `-DCMAKE_INSTALL_PREFIX=<...>` to `cmake`.

### Build the test suite

The test suite is built by default unless `-DENABLE_TEST=OFF` is passed to `cmake`. Run the tests with the following command from within `build`.

```shell-session
$ ctest -V
```

### Build the Python binding

To install the `xchainer` Python package, run `python setup.py install` at the repository root.

To build the Python binding as a C++ project, pass `-DBUILD_PYTHON=1` to `cmake`,
then `make` will automatically build the Python binding.

## Information for developers

### clang-format

zsh:

```shell-session
clang-format -i xchainer/**/*.{cc,h}
```

bash:

```shell-session
find xchainer \( -name '*.cc' -o -name '*.h' \) -type f -print0 | xargs -0 clang-format -i
```

### C++ Coverage

We use gcov to measure C++ code coverage.
Pass `-DENABLE_COVERAGE=ON` to cmake to enable gcov, and build Python package in Debug mode as:

```
python setup.py build --debug --build-temp ./build --build-lib ./build develop
( cd build && cmake -DENABLE_COVERAGE=ON .. && make )
```

Run both Python and C++ test suite:

```
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
