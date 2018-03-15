FROM chainer/chainer:latest

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN pip install asv virtualenv Cython
