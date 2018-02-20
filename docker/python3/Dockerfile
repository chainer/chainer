FROM nvidia/cuda:8.0-cudnn6-devel

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN pip3 install cupy==4.0.0b3 chainer==4.0.0b3
