Export ONNX and Compile it with NNVM
====================================

## Requirements

### NNVM>=0.8.0

```bash
git clone --recursive https://github.com/dmlc/nnvm
cd nnvm && mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=../install ..
make -j$(nproc) install
cd ../python && python setup.py install
```

### TVM>=0.1.0

```bash
git clone --recursive https://github.com/dmlc/tvm
cd tvm && mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=../install ..
make -j$(nproc) install
cd ../python && python setup.py install
```

### TOPI>=0.1.0

```
cd tvm/topi
python setup.py install
```

### SciPy

```
pip install scipy
```