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

export CHAINER_CI=flexci

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
  CHAINER_BUILD_CHAINERX=1 CHAINERX_BUILD_CUDA=1 \
  MAKEFLAGS="-j$(get_build_concurrency)" \
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
  nice -n 19 make "-j$(get_build_concurrency)"
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
      --retry=1
  )
  if [ "${SPREADSHEET_ID:-}" != '' ]; then
    xpytest_args+=(--spreadsheet_id="${SPREADSHEET_ID}")
  fi
  # TODO(niboshi): Allow option pass-through (https://github.com/chainer/xpytest/issues/14)
  export PYTEST_ADDOPTS=-rfEX
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
# Despite the name, Python 2.7 is no longer tested in the master branch.
# TODO(niboshi): Completely remove Python 2.7 after discontinuing v6 series.
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
  # Install Chainer for python3.5.
  CHAINER_BUILD_CHAINERX=1 CHAINERX_BUILD_CUDA=1 \
  MAKEFLAGS="-j$(get_build_concurrency)" \
  CHAINERX_NVCC_GENERATE_CODE=arch=compute_70,code=sm_70 \
      python3.5 -m pip install /chainer[test] 2>&1 >/tmp/install-py35.log &
  install_pid=$!

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
      --retry=1
  )
  if [ "${SPREADSHEET_ID:-}" != '' ]; then
    xpytest_args+=(--spreadsheet_id="${SPREADSHEET_ID}")
  fi
  # TODO(niboshi): Allow option pass-through (https://github.com/chainer/xpytest/issues/14)
  export PYTEST_ADDOPTS=-rfEX
  # NOTE: PYTHONHASHSEED=0 is necessary to use pytest-xdist.
  OMP_NUM_THREADS=1 PYTHONHASHSEED=0 xpytest "${xpytest_args[@]}" \
      '/chainer/tests/chainer_tests/**/test_*.py' && :
  py35_test_status=$?

  #-----------------------------------------------------------------------------
  # Finalize
  #-----------------------------------------------------------------------------
  echo "py35_test_status=${py35_test_status}"
  exit ${py35_test_status}
}

# test_chainermn is a test function for chainermn
test_chainermn() {
  export PYENV_VERSION=""
  . /root/.bash_profile

  TEST_PYTHON_VERSIONS="3.6.8"
  ret=0
  for VERSION in $TEST_PYTHON_VERSIONS
  do
    pyenv shell ${VERSION}
	MAJOR_VERSION=${VERSION:0:1}
	test_chainermn_sub
	tmp_ret=$?
	ret=$(( ret || tmp_ret ))
  done
  exit $ret
}

# test_chainermn_sub runs tests for chainermn with current Python runtime
test_chainermn_sub() {
  marker='not slow'
  if (( !GPU )); then
    marker+=' and not gpu'
  else
    marker+=' and gpu'
  fi

  #-----------------------------------------------------------------------------
  # Install CuPy from wheel
  #-----------------------------------------------------------------------------
  pip install /cupy-wheel/cupy-*-cp${MAJOR_VERSION}*-cp${MAJOR_VERSION}*-linux_x86_64.whl

  #-----------------------------------------------------------------------------
  # Install Chainer
  #-----------------------------------------------------------------------------
  CHAINER_BUILD_CHAINERX=1 CHAINERX_BUILD_CUDA=1 \
  MAKEFLAGS="-j$(get_build_concurrency)" \
  CHAINERX_NVCC_GENERATE_CODE=arch=compute_70,code=sm_70 \
      python -m pip install /chainer[test] 2>&1 >/tmp/install-py3.log &
  install_pid=$!

  if ! wait $install_pid; then
    cat /tmp/install-py3.log
    exit 1
  fi

  #-----------------------------------------------------------------------------
  # Test python
  #-----------------------------------------------------------------------------
  mpirun --allow-run-as-root -n 2 python -m pytest --color=yes \
                   --full-trace \
                   --durations=10 \
                   -x --capture=no \
                   -s -v -m "${marker}" \
				   /chainer/tests/chainermn_tests
}

# get_build_concurrency determines the parallelism of the build process.
# Currently maximum is set to 16 to avoid exhausting memory.
get_build_concurrency() {
    local num_cores="$(nproc)"
    local num_cores_max="16"
    if [ ${num_cores} -gt ${num_cores_max} ]; then
        echo "${num_cores_max}"
    else
        echo "${num_cores}"
    fi
}

################################################################################
# Bootstrap
################################################################################
case "${TARGET}" in
  'py37' ) test_py37;;
  'py27and35' ) test_py27and35;;
  'chainermn' ) test_chainermn;;
  * ) echo "Unsupported target: ${TARGET}" >&2; exit 1;;
esac
