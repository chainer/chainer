FROM ubuntu:16.04


RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
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
    python3-dev \
    python3-numpy \
    python3-pip \
    python3-setuptools \
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

RUN pip3 install --upgrade pip setuptools

RUN pip3 install --default-timeout=10000 wheel \
    six \
    h5py \
    nose \
    protobuf \
    Pillow \
    filelock \
    distribute


RUN git clone -b master_v3 https://github.com/intel/chainer && \
    cd chainer  && \
    python3 setup.py install

