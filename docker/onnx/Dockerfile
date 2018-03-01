FROM ubuntu:trusty

MAINTAINER Shunta Saito <shunta.saito@gmail.com>

RUN apt-get update
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git

RUN curl -L https://repo.continuum.io/miniconda/Miniconda3-4.3.21-Linux-x86_64.sh -o miniconda.sh
RUN bash miniconda.sh -b -p miniconda
ENV PATH /miniconda/bin:$PATH
ENV LD_LIBRARY_PATH /miniconda/lib:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH /miniconda/lib:$LIBRARY_PATH
ENV CPATH /miniconda/include:$CPATH

RUN conda config --set always_yes yes --set changeps1 no
RUN conda install -c conda-forge protobuf
RUN conda install -c conda-forge glog
RUN conda install -c conda-forge numpy 
RUN conda install -c conda-forge scipy 
RUN conda install -c conda-forge cmake
RUN conda install -c conda-forge gflags
RUN conda install -c conda-forge pybind11

RUN pip install chainer==3.1.0
RUN pip install future
RUN pip install hypothesis
RUN pip install flake8
RUN pip install hacking
RUN pip install autopep8
RUN pip install mock
RUN pip install pytest
RUN pip install six
RUN pip install onnx==1.0.0

RUN git clone --recursive https://github.com/caffe2/caffe2.git $HOME/caffe2
RUN cd $HOME/caffe2 && mkdir build && cd build \
    && cmake .. \
    -DUSE_CUDA=OFF \
    -DUSE_NCCL=OFF \
    -DUSE_NNPACK=OFF \
    -DUSE_ROCKSDB=OFF \
    -DUSE_LEVELDB=OFF \
    -DUSE_LMDB=OFF \
    -DUSE_OPENCV=OFF \
    -DBUILD_TEST=OFF \
    -DBUILD_BENCHMARK=OFF \
    -DPYTHON_EXECUTABLE=/miniconda/bin/python \
    -DPYTHON_INCLUDE_DIR=/miniconda/include \
    -DPYTHON_INCLUDE_DIR2=/miniconda/include/python3.6m \
    -DPYTHON_LIBRARY=/miniconda/lib/libpython3.6m.so \
    -DPYTHON_LIBRARY_DEBUG=/miniconda/lib/libpython3.6m.so \
    -DPROTOBUF_INCLUDE_DIR=/miniconda/include \
    -DPROTOBUF_LIBRARY=/miniconda/lib/libprotobuf.so \
    -DPROTOBUF_LIBRARY_DEBUG=/miniconda/lib/libprotobuf.so \
    -DPROTOBUF_LITE_LIBRARY=/miniconda/lib/libprotobuf-lite.so \
    -DPROTOBUF_LITE_LIBRARY_DEBUG=/miniconda/lib/libprotobuf-lite.so \
    -DPROTOBUF_PROTOC_LIBRARY=/miniconda/lib/libprotoc.so \
    -DPROTOBUF_PROTOC_LIBRARY_DEBUG=/miniconda/lib/libprotoc.so \
    -Dpybind11_INCLUDE_DIR=/miniconda/include/python3.6m
RUN cd $HOME/caffe2/build && \
    LDFLAGS="-L/miniconda/lib" LD_LIBRARY_PATH="/miniconda/lib" \
    CFLAGS="-I/miniconda/include" CPATH="/miniconda/include" \
    make -j"$(nproc)" install
ENV PYTHONPATH /usr/local/
RUN python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
RUN pip install onnx-caffe2==1.0.0

# Install MXNet
RUN pip install mxnet
RUN cd $HOME && git clone https://github.com/onnx/onnx-mxnet.git && cd onnx-mxnet && python setup.py install

# Install NNVM
RUN git clone --recursive https://github.com/dmlc/nnvm $HOME/nnvm
RUN cd $HOME/nnvm && mkdir build && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX=../install ..
RUN cd $HOME/nnvm/build && make -j$(nproc) install
RUN cd $HOME/nnvm/python && python setup.py install

# Install TVM
RUN git clone --recursive https://github.com/dmlc/tvm $HOME/tvm
RUN cd $HOME/tvm && mkdir build && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX=../install .. \
    && make -j$(nproc) install
RUN cd $HOME/tvm/python && python setup.py install
RUN cd $HOME/tvm/topi/python && python setup.py install