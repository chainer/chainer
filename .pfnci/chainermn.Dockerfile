ARG BASE_IMAGE=9.2-cudnn7-devel

FROM nvidia/cuda:${BASE_IMAGE}

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    git \
    cmake \
    libblas3 \
    libblas-dev \
	curl \
	zlib1g-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
	libffi-dev \
	build-essential \
    libbz2-dev \
	ssh \
	wget \
    && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Install OpenMPI with CUDA
ENV OMPI_VERSION 3.1.3

RUN cd /tmp && wget -q https://www.open-mpi.org/software/ompi/v${OMPI_VERSION%\.*}/downloads/openmpi-$OMPI_VERSION.tar.bz2 && \
  tar -xjf openmpi-$OMPI_VERSION.tar.bz2

RUN cd /tmp/openmpi-$OMPI_VERSION && \
    ./configure --prefix=/usr --with-cuda --disable-oshmem --disable-mpi-java --disable-java --disable-mpi-fortran && \
    make -j 10 && make install && cd && rm -r /tmp/openmpi-$OMPI_VERSION* && \
    /usr/bin/ompi_info --parsable --all | grep -q "mpi_built_with_cuda_support:value:true" && \
    rm -rf /tmp/openmpi-*


# Install pyenv

ENV PYENV_ROOT /usr/local/pyenv
ENV BASH_PROFILE /root/.bash_profile

RUN git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT

RUN touch $BASH_PROFILE && \
    echo 'export PYENV_ROOT="$PYENV_ROOT"' >> $BASH_PROFILE && \
	echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> $BASH_PROFILE && \
	echo 'eval "$(pyenv init -)"' >> $BASH_PROFILE

#
# Install Python and necessary packages
#


# Python 3.6.8
ENV PYTHON_VERSION 3.6.8
COPY . /cupy
RUN . $BASH_PROFILE && cd /cupy && pyenv install $PYTHON_VERSION && \
	pyenv shell ${PYTHON_VERSION} && \
	pip install -U pip && \
	pip install cython && \
	pip install chainer pytest mock mpi4py && \
	pip uninstall -y chainer && \
        pip install .
