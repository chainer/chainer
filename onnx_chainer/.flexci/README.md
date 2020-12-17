**Build scripts**

```bash
$ export BUILD_PY_VER=37
$ docker build -t kmaehashi/onnx-chainer:ci-py${BUILD_PY_VER} --build-arg PYTHON_VERSION=${BUILD_PY_VER} .
```
