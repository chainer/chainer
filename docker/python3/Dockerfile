FROM nvidia/cuda:7.5-cudnn5-devel

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

<<<<<<< HEAD
RUN pip3 install cupy==1.0.0b1 chainer==2.0.0b1
=======
RUN pip3 install chainer==1.24.0
>>>>>>> origin/master
