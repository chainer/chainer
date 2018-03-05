# Docker image on which xChaier's CircleCI jobs are based
FROM ubuntu:xenial
ENV HOME /root
ENV PATH "${HOME}/miniconda/bin:${PATH}"

# Setup

RUN apt-get update
RUN apt-get install -y g++ cmake clang clang-format clang-tidy git python3-dev lcov wget
RUN apt-get install openssh-client  # To avoid warning on checkout
RUN apt-get install -y bzip2  # To install miniconda

# Setup conda

RUN mkdir -p downloads
RUN wget 'https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh' -O downloads/miniconda.sh
RUN bash downloads/miniconda.sh -b -p "$HOME"/miniconda
RUN conda config --set changeps1 no
RUN conda update -y -q conda
RUN conda create -y -q --name testenv python=3.6 pip
