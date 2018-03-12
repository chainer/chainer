#!/bin/bash -uex

function run_asv() {
  CHAINER_COMMIT="${1}"; shift
  CUPY_COMMIT="${1}"; shift

  # Clone CuPy.
  if [ ! -d cupy ]; then
    git clone https://github.com/cupy/cupy.git
  fi

  # Build CuPy commit to use for benchmark.
  # Note that CuPy will be injected from current environment via `PYTHONPATH`
  # instead of `matrix` in `asv.conf.json`, because Chainer and CuPy are
  # tightly-coupled that we should manually pick which commit of CuPy to use.
  # The version of the python command in outer world must match with the
  # version used in the benchmark virtualenv.
  pushd cupy
  git remote update
  git clean -fdx
  git checkout "$(git show --format="%H" ${CUPY_COMMIT})"
  python setup.py build_ext --inplace
  export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
  popd

  # Run the benchmark.
  # Uncomment the following lines to diagnose installation issues.
  #export PIP_VERBOSE=True
  #export PIP_LOG=pip.log
  asv run --step 1 "$@" "${CHAINER_COMMIT}"
}

run_asv "$@"
