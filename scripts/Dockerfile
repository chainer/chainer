# Dockerfile for running chainer (https://github.com/pfnet/chainer)
# Usage:
#   > sudo docker run -t -i kazunori279/chainer
#   # cd chainer/examples/mnist
#   # python download_convert.py
#   # python train_mnist.py

FROM scivm/scientific-python-2.7 
MAINTAINER kazunori279 "https://github.com/kazunori279" 

# basic setup 
ENV DEBIAN_FRONTEND noninteractive
RUN echo "deb http://archive.ubuntu.com/ubuntu precise main universe" > /etc/apt/sources.list
RUN apt-get update
RUN apt-get upgrade -y --force-yes

# install git
RUN apt-get install -y --force-yes curl git
RUN mkdir /git && cd /git && git init

# install chainer
RUN pip install six chainer
RUN git clone https://github.com/pfnet/chainer.git

# bash
ENTRYPOINT ["/bin/bash"]
