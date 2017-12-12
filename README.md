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

### Build the Python binding

To install the `xchainer` Python package, run `python setup.py install` at the repository root.

To build the Python binding as a C++ project, pass `-DBUILD_PYTHON=1` to `cmake`,
then `make` will automatically build the Python binding.

## Information for developers

### clang-format

zsh:

```shell-session
clang-format -i **/*.{cc,h}
```

bash:

```shell-session
find . \( -name '*.cc' -o -name '*.h' \) -type f -print0 | xargs -0 clang-format -i
```
