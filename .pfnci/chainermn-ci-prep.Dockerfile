ARG BASE_IMAGE

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


# Install pyenv and pyenv-virtualenv

ENV PYENV_ROOT /usr/local/pyenv
ENV BASH_PROFILE /root/.bash_profile

RUN git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT

RUN touch $BASH_PROFILE && \
    echo 'export PYENV_ROOT="$PYENV_ROOT"' >> $BASH_PROFILE && \
	echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> $BASH_PROFILE && \
	echo 'eval "$(pyenv init -)"' >> $BASH_PROFILE
	
RUN . $BASH_PROFILE && \
	git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv


# Clone CuPy

ENV CUPY_MASTER_ROOT /usr/local/cupy-master
RUN git clone https://github.com/cupy/cupy.git $CUPY_MASTER_ROOT --depth 1
ENV CUPY_V6_ROOT /usr/local/cupy-v6
RUN git clone https://github.com/cupy/cupy.git $CUPY_V6_ROOT -b v6 --depth 1

#
# Install different versions of python and necessary packages
#

# Python 2.7.16

ENV PYTHON_VERSION 2.7.16

RUN . $BASH_PROFILE && pyenv install $PYTHON_VERSION

RUN . $BASH_PROFILE && \
    pyenv virtualenv $PYTHON_VERSION ${PYTHON_VERSION}-cupy-master && \
	pyenv shell ${PYTHON_VERSION}-cupy-master && \
	pip install -U pip && \
	cd $CUPY_MASTER_ROOT && \
	pip install cython && \
	pip install . && \
	pip install chainer pytest mock mpi4py

RUN . $BASH_PROFILE && \
    pyenv virtualenv $PYTHON_VERSION ${PYTHON_VERSION}-cupy-v6 && \
	pyenv shell ${PYTHON_VERSION}-cupy-v6 && \
	pip install -U pip && \
	cd $CUPY_V6_ROOT && \
	pip install cython && \
	pip install . && \
	pip install chainer pytest mock mpi4py


# Python 3.5.7

ENV PYTHON_VERSION 3.5.7

RUN . $BASH_PROFILE && pyenv install $PYTHON_VERSION

RUN . $BASH_PROFILE && \
    pyenv virtualenv $PYTHON_VERSION ${PYTHON_VERSION}-cupy-master && \
	pyenv shell ${PYTHON_VERSION}-cupy-master && \
	pip install -U pip && \
	cd $CUPY_MASTER_ROOT && \
	pip install cython && \
	pip install . && \
	pip install chainer pytest mock mpi4py

RUN . $BASH_PROFILE && \
    pyenv virtualenv $PYTHON_VERSION ${PYTHON_VERSION}-cupy-v6 && \
	pyenv shell ${PYTHON_VERSION}-cupy-v6 && \
	pip install -U pip && \
	cd $CUPY_V6_ROOT && \
	pip install cython && \
	pip install . && \
	pip install chainer pytest mock mpi4py


# Python 3.7.3

ENV PYTHON_VERSION 3.7.3

RUN . $BASH_PROFILE && pyenv install $PYTHON_VERSION

RUN . $BASH_PROFILE && \
    pyenv virtualenv $PYTHON_VERSION ${PYTHON_VERSION}-cupy-master && \
	pyenv shell ${PYTHON_VERSION}-cupy-master && \
	pip install -U pip && \
	cd $CUPY_MASTER_ROOT && \
	pip install cython && \
	pip install . && \
	pip install chainer pytest mock mpi4py

RUN . $BASH_PROFILE && \
    pyenv virtualenv $PYTHON_VERSION ${PYTHON_VERSION}-cupy-v6 && \
	pyenv shell ${PYTHON_VERSION}-cupy-v6 && \
	pip install -U pip && \
	cd $CUPY_V6_ROOT && \
	pip install cython && \
	pip install . && \
	pip install chainer pytest mock mpi4py
