FROM nvidia/cuda:10.1-cudnn7-devel

LABEL author="Daisuke Tanaka <duaipp@gmail.com>, Shunta Saito <shunta.saito@gmail.com>"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    curl \
    wget \
    git \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libffi-dev && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

WORKDIR /root

# Install pyenv
RUN curl -L -O https://github.com/pyenv/pyenv/archive/v1.2.9.tar.gz && \
    tar zxf v1.2.9.tar.gz && rm -rf v1.2.9.tar.gz && \
    mv pyenv-1.2.9 .pyenv

ENV PYENV_ROOT=/root/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN eval "$(pyenv init -)"

ARG PYTHON_VERSION

RUN if [ ${PYTHON_VERSION} = "37" ]; then \
        CONFIGURE_OPTS="--enable-shared" pyenv install 3.7.2; \
        pyenv global 3.7.2; pyenv rehash; \
    elif [ ${PYTHON_VERSION} = "36" ]; then \
        CONFIGURE_OPTS="--enable-shared" pyenv install 3.6.8; \
        pyenv global 3.6.8; pyenv rehash; \
    elif [ ${PYTHON_VERSION} = "35" ]; then \
        CONFIGURE_OPTS="--enable-shared" pyenv install 3.5.6; \
        pyenv global 3.5.6; pyenv rehash; \
    fi

ENV CUDA_PATH=/usr/local/cuda
ENV PATH=$CUDA_PATH/bin:$PATH
ENV CPATH=$CUDA_PATH/include:/usr/local/include:$CPATH
ENV LD_LIBRARY_PATH=$CUDA_PATH/lib64:$CUDA_PATH/lib:/usr/local/lib:$LD_LIBRARY_PATH
