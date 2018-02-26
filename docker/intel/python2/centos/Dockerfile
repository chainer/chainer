FROM centos:7

USER root

RUN yum -y install epel-release &&\
    yum upgrade -y

RUN yum install -y \
        redhat-rpm-config \
        cmake \
        git \
        wget \
        ssh \
        gcc-c++ \
        boost-devel \
        gflags-devel \
        glog-devel \
        hdf5-devel \
        opencv-devel \
        protobuf-devel \
        liblapack-dev \
        libssl-dev \
        protobuf-compiler \
        python-devel \
        python-numpy \
        python-pip \
        python-setuptools \
        openssl \
        gcc-gfortran \
        pcre \
        pcre-devel

RUN yum clean all

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

RUN pip install --default-timeout=10000 six \
    h5py \
    nose \
    protobuf \
    Pillow \
    filelock

RUN pip3 install chainer==4.0.0b4
