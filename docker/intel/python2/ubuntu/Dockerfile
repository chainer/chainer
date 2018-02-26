FROM ubuntu:16.04

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    vim \
    ssh \
    libboost-all-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libhdf5-serial-dev \
    libopencv-dev \
    libprotobuf-dev \
    liblapack-dev \
    libssl-dev \
    protobuf-compiler \
    python-dev \
    python-numpy \
    python-pip \
    python-setuptools \
    openssl \
    curl && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN export LD_LIBRARY_PATH="$HOME/.chainer/lib:$HOME/.local/lib:/usr/local/lib" && \
    export LIBRARY_PATH="$LD_LIBRARY_PATH"

RUN  mkdir Downloads && \
     cd Downloads && \
     wget https://nchc.dl.sourceforge.net/project/swig/swig/swig-3.0.12/swig-3.0.12.tar.gz && \
     tar -xf swig-3.0.12.tar.gz && \
     cd swig-3.0.12/ && \
     ./configure && \
     make -j && \
     make install && \
     cd ../../

RUN pip install --upgrade pip setuptools

RUN pip install --default-timeout=10000 wheel \
    six \
    h5py \
    nose \
    protobuf \
    Pillow \
    filelock \
    distribute

RUN pip3 install chainer==4.0.0b4
