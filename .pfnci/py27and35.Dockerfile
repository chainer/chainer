FROM golang AS xpytest
RUN git clone --depth=1 https://github.com/chainer/xpytest.git /xpytest
RUN cd /xpytest && \
    go build -o /usr/local/bin/xpytest ./cmd/xpytest

FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python-dev python-pip python-wheel python-setuptools \
    python3-dev python3-pip python3-wheel python3-setuptools \
    wget git g++ make cmake libblas3 libblas-dev liblapack-dev && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN python3.5 -m pip install --upgrade pip setuptools

RUN python3.5 -m pip install \
    'cython>=0.28.0' 'ideep4py<2.1' 'pytest==4.1.1' 'pytest-xdist==1.26.1' \
    mock setuptools typing \
    typing_extensions filelock 'numpy>=1.9.0' 'protobuf>=3.0.0' 'six>=1.9.0'

COPY --from=xpytest /usr/local/bin/xpytest /usr/local/bin/xpytest

COPY . /cupy
RUN cd /cupy && \
    echo 'install-%:\n\t$* -m pip install .\n' > /tmp/install.mk && \
    cat /tmp/install.mk && \
    make -f /tmp/install.mk -j 2 install-python3.5
