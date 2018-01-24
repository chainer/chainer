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
        protobuf-devel \
        liblapack-dev \
        libssl-dev \
        protobuf-compiler \
        openssl \
        openssl-devel \
        bzip2-devel \
        expat-devel \
        gdbm-devel \
        readline-devel \
        sqlite-devel \
        gcc-gfortran \
        pcre \
        pcre-devel \

RUN yum clean all


RUN mkdir Downloads && \
    cd Downloads && \
    wget https://www.python.org/ftp/python/3.5.2/Python-3.5.2.tgz && \
    tar -zxvf Python-3.5.2.tgz && \
    cd Python-3.5.2 && \
    ./configure --prefix=/usr/local/python3.5 --enable-optimizations && \
    make -j && \
    make install && \
    ln -s /usr/local/python3.5/bin/python3 /usr/bin/python3 && \
    ln -s /usr/local/python3.5/bin/pip3.5 /usr/bin/pip3 && \
    cd ../

RUN export LD_LIBRARY_PATH="$HOME/.chainer/lib:$HOME/.local/lib:/usr/local/lib" && \
    export LIBRARY_PATH="$LD_LIBRARY_PATH"


RUN wget https://nchc.dl.sourceforge.net/project/swig/swig/swig-3.0.12/swig-3.0.12.tar.gz && \
     tar -xf swig-3.0.12.tar.gz && \
     cd swig-3.0.12/ && \
     ./configure && \
     make -j && \
     make install && \
     cd ../../


RUN pip3 install --default-timeout=10000 six \
    h5py \
    nose \
    protobuf \
    Pillow \
    filelock


RUN git clone -b master_v3 https://github.com/intel/chainer && \
    cd chainer && \
    python3 setup.py install
