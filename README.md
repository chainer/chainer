# xChainer

## Build instruction

Build `build/xchainer/libxchainer.so` with the following commands.

```shell-session
$ mkdir build
$ cd build
$ cmake ..
$ make
```

Install headers and the library with `make install`.
To specify the installation path, pass `-DCMAKE_INSTALL_PREFIX=<...>` to `cmake`.

### clang-format

zsh:

```shell-session
clang-format -i **/*.{cc,h}
```

bash:

```shell-session
find . \( -name '*.cc' -o -name '*.h' \) -type f -print0 | xargs -0 clang-format -i
```
