#!/bin/bash
# run.sh is a script to run unit tests inside Docker.  This should be injected
# by and called from script.sh.
#
# Usage: run.sh [target]
# - target is a test target (e.g., "py37").
#
# Environment variables:
# - GPU (default: 0) ... Set a number of GPUs to GPU.
#       CAVEAT: Setting GPU>0 disables non-GPU tests, and setting GPU=0 disables
#               GPU tests.
# - SPREADSHEET_ID ... Set SPREADSHEET_ID (e.g.,
#       "1u5OYiPOL3XRppn73XBSgR-XyDuHKb_4Ilmx1kgJfa-k") to enable xpytest to
#       report to a spreadsheet.

set -eux

cp -a /src /chainer
cp /chainer/setup.cfg /
cd /

# Remove pyc files.  When the CI is triggered with a user's local files, pyc
# files generated on the user's local machine and they often cause failures.
find /chainer -name "*.pyc" -exec rm -f {} \;

TARGET="$1"
: "${GPU:=0}"
: "${XPYTEST_NUM_THREADS:=$(nproc)}"

# Use multi-process service to prevent GPU flakiness caused by running many
# processes on a GPU.  Specifically, it seems running more than 16 processes
# sometimes causes "cudaErrorLaunchFailure: unspecified launch failure".
if (( GPU > 0 )); then
    nvidia-smi -c EXCLUSIVE_PROCESS
    nvidia-cuda-mps-control -d
fi

################################################################################
# Test functions
################################################################################

# test_py37 is a test function for chainer.py37.
test_py37() {
  #-----------------------------------------------------------------------------
  # Configure parameters
  #-----------------------------------------------------------------------------
  export CHAINERX_TEST_CUDA_DEVICE_LIMIT="${GPU}"
  marker='not slow'
  if (( !GPU )); then
    marker+=' and not (gpu or cudnn or cuda)'
    bucket=1
  else
    marker+=' and (gpu or cudnn or cuda)'
    bucket="${GPU}"
  fi

  #-----------------------------------------------------------------------------
  # Install Chainer
  #-----------------------------------------------------------------------------
  CHAINER_BUILD_CHAINERX=1 CHAINERX_BUILD_CUDA=1 MAKEFLAGS="-j$(nproc)" \
  CHAINERX_NVCC_GENERATE_CODE=arch=compute_70,code=sm_70 \
      python3.7 -m pip install /chainer[test] 2>&1 >/tmp/install.log &
  install_pid=$!

  #-----------------------------------------------------------------------------
  # ChainerX C++ tests
  #-----------------------------------------------------------------------------
  mkdir -p /root/chainerx_build
  pushd /root/chainerx_build
  cmake \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCHAINERX_BUILD_CUDA="$(
          if (( GPU )); then echo 'ON'; else echo 'OFF'; fi
      )" \
      -DCHAINERX_BUILD_TEST=ON \
      -DCHAINERX_BUILD_PYTHON=OFF \
      -DCHAINERX_WARNINGS_AS_ERRORS=ON \
      /chainer/chainerx_cc
  # NOTE: Use nice to prioritize pip install process.
  nice -n 19 make "-j$(nproc)"
  ctest --output-on-failure "-j$(nproc)" && :
  cc_test_status=$?
  popd

  #-----------------------------------------------------------------------------
  # Chainer/ChainerX python tests
  #-----------------------------------------------------------------------------
  if ! wait $install_pid; then
    cat /tmp/install.log
    exit 1
  fi
  xpytest_args=(
      --python=python3.7 -m "${marker}"
      --bucket="${bucket}" --thread="$(( XPYTEST_NUM_THREADS / bucket ))"
      --hint="/chainer/.pfnci/hint.pbtxt"
  )
  if [ "${SPREADSHEET_ID:-}" != '' ]; then
    xpytest_args+=(--spreadsheet_id="${SPREADSHEET_ID}")
  fi
  # TODO(imos): Enable xpytest to support python_files setting in setup.cfg.
  OMP_NUM_THREADS=1 xpytest "${xpytest_args[@]}" \
      '/chainer/tests/chainerx_tests/**/test_*.py' \
      '/chainer/tests/chainer_tests/**/test_*.py' \
      && :
  py_test_status=$?

  #-----------------------------------------------------------------------------
  # Finalize
  #-----------------------------------------------------------------------------
  echo "cc_test_status=${cc_test_status}"
  echo "py_test_status=${py_test_status}"
  exit $((cc_test_status || py_test_status))
}

# test_py27and35 is a test function for chainer.py27and35.
test_py27and35() {
  #-----------------------------------------------------------------------------
  # Configure parameters
  #-----------------------------------------------------------------------------
  export CHAINERX_TEST_CUDA_DEVICE_LIMIT="${GPU}"
  marker='not slow'
  if (( !GPU )); then
    marker+=' and not (gpu or cudnn or cuda)'
    bucket=1
  else
    marker+=' and (gpu or cudnn or cuda)'
    bucket="${GPU}"
  fi

  #-----------------------------------------------------------------------------
  # Install Chainer
  #-----------------------------------------------------------------------------
  # Install Chainer for python2.7.
  if ! python2.7 -m pip install /chainer[test] 2>&1 >/tmp/install-py27.log; then
    cat /tmp/install-py27.log
    exit 1
  fi
  # Install Chainer for python3.5 asynchronously.
  # NOTE: Installation of python3.5 takes much longer time because it requires
  # ChainerX builds.  It is difficult to speed up with parallelization, so this
  # script runs it in the background of python2.7 unit testing.
  CHAINER_BUILD_CHAINERX=1 CHAINERX_BUILD_CUDA=1 MAKEFLAGS="-j$(nproc)" \
  CHAINERX_NVCC_GENERATE_CODE=arch=compute_70,code=sm_70 \
      python3.5 -m pip install /chainer[test] 2>&1 >/tmp/install-py35.log &
  install_pid=$!

  #-----------------------------------------------------------------------------
  # Test python2.7
  #-----------------------------------------------------------------------------
  xpytest_args=(
      --python=python2.7 -m "${marker}"
      --bucket="${bucket}" --thread="$(( XPYTEST_NUM_THREADS / bucket ))"
      --hint="/chainer/.pfnci/hint.pbtxt"
  )
  if [ "${SPREADSHEET_ID:-}" != '' ]; then
    xpytest_args+=(--spreadsheet_id="${SPREADSHEET_ID}")
  fi
  OMP_NUM_THREADS=1 xpytest "${xpytest_args[@]}" \
      '/chainer/tests/chainer_tests/**/test_*.py' && :
  py27_test_status=$?

  #-----------------------------------------------------------------------------
  # Test python3.5
  #-----------------------------------------------------------------------------
  if ! wait $install_pid; then
    cat /tmp/install-py35.log
    exit 1
  fi
  xpytest_args=(
      --python=python3.5 -m "${marker}"
      --bucket="${bucket}" --thread="$(( XPYTEST_NUM_THREADS / bucket ))"
      --hint="/chainer/.pfnci/hint.pbtxt"
  )
  if [ "${SPREADSHEET_ID:-}" != '' ]; then
    xpytest_args+=(--spreadsheet_id="${SPREADSHEET_ID}")
  fi
  # NOTE: PYTHONHASHSEED=0 is necessary to use pytest-xdist.
  OMP_NUM_THREADS=1 PYTHONHASHSEED=0 xpytest "${xpytest_args[@]}" \
      '/chainer/tests/chainer_tests/**/test_*.py' && :
  py35_test_status=$?

  #-----------------------------------------------------------------------------
  # Finalize
  #-----------------------------------------------------------------------------
  echo "py27_test_status=${py27_test_status}"
  echo "py35_test_status=${py35_test_status}"
  exit $((py27_test_status || py35_test_status))
}

################################################################################
# Bootstrap
################################################################################
case "${TARGET}" in
  'py37' ) test_py37;;
  'py27and35' ) test_py27and35;;
  * ) echo "Unsupported target: ${TARGET}" >&2; exit 1;;
esac
