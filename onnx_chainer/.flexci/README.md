**Build scripts**

```bash
$ export BUILD_PY_VER=37
$ docker build -t disktnk/onnx-chainer:ci-py${BUILD_PY_VER} -f .flexci/Dockerfile --build-arg PYTHON_VERSION=${BUILD_PY_VER} .
```
