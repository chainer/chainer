FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04

LABEL chainerx_test_image=1

ARG TEMP_DIR=/tmp/tmp-xchainer-ci-dockerfile
ARG WORKSPACE_DIR
ARG WORK_DIR
ARG CONDA_DIR
RUN mkdir -p "$TEMP_DIR"


ADD setup-ubuntu.sh "$TEMP_DIR"/
ADD setup-conda.sh "$TEMP_DIR"/

# Install apt packages
RUN bash "$TEMP_DIR"/setup-ubuntu.sh

# Create directories
ARG USER_TEMP_DIR="$TEMP_DIR"/user
RUN mkdir -p "$USER_TEMP_DIR"
RUN mkdir -p "$WORKSPACE_DIR"

# Add a user for running test.
# {{{UID}}} will be replaced by the host UID.
RUN useradd -m testuser -u {{{UID}}}

RUN chown testuser:testuser "$USER_TEMP_DIR"
RUN chown testuser:testuser "$WORKSPACE_DIR"

USER testuser
WORKDIR "$USER_TEMP_DIR"

RUN mkdir -p "$WORK_DIR"
RUN mkdir -p "$CONDA_DIR"

# Install conda
RUN bash "$TEMP_DIR"/setup-conda.sh "$USER_TEMP_DIR"/conda "$CONDA_DIR"

# Install cupy
RUN bash -c 'source '"$CONDA_DIR"'/bin/activate testenv && pip install -U cython && pip install git+https://github.com/cupy/cupy@v7'
