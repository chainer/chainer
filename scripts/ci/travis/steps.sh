#!/usr/bin/env bash
# This script defines the test steps, used from run-step.sh.
# In each step, the entire script is sourced first and subsequently one of the step functions is invoked.
# Therefore, the global part of the script, outside the function defnitions, is run before every single steps.

# TODO(niboshi): Definitions of the steps could be merged with scripts/ci/steps.sh.

# Check preconditions
test -f "$CHAINER_BASH_ENV"
test -d "$REPO_DIR"
test -d "$WORK_DIR"


DEFAULT_JOBS=2


_install_chainer_deps() {
    # Install extras_require of Chainer.
    local extras="$1"

    # It's not possible to install only requirements.
    # Chainer is uninstalled after the installation.
    # TODO(niboshi): Use other installation tool
    # On Windows pip does not seem to support installing extras with full path.
    pushd "$REPO_DIR"
    pip install -e .["$extras"]
    popd
    pip uninstall -y chainer
}


step_install_chainer_style_check_deps() {
    _install_chainer_deps stylecheck
}


step_install_chainer_test_deps() {
    _install_chainer_deps test

    # Optional packages
    local reqs=(
        theano
        h5py
        pillow
    )
    pip install "${reqs[@]}"
}


step_before_install_chainer_test() {
    # Remove oclint as it conflicts with GCC (indirect dependency of hdf5)
    if [[ $TRAVIS_OS_NAME = "osx" ]]; then
        brew update >/dev/null
        brew uninstall openssl@1.1 || :  # tentative workaround: pyenv/pyenv#1302
        brew outdated pyenv || brew upgrade pyenv

        PYTHON_CONFIGURE_OPTS="--enable-unicode=ucs2" pyenv install -ks $PYTHON_VERSION
        pyenv global $PYTHON_VERSION
        python --version

        brew install hdf5
    fi
}

step_before_install_chainermn_test_deps() {
    case "$TRAVIS_OS_NAME" in
        linux)
            local pkgs=(openmpi-bin openmpi-common libopenmpi-dev)
            sudo apt-get install -y "${pkgs[@]}"
            ;;
        osx)
            brew install open-mpi
            ;;
        *)
            false
            ;;
    esac
}


step_install_chainermn_test_deps() {
    pip install mpi4py
}


step_install_chainer_docs_deps() {
    _install_chainer_deps docs
}


step_python_style_check() {
    check_targets=(
        "$REPO_DIR"/*.py
        "$REPO_DIR"/chainer
        "$REPO_DIR"/chainermn
        "$REPO_DIR"/chainerx
        "$REPO_DIR"/tests
        "$REPO_DIR"/examples
        "$REPO_DIR"/chainerx_cc/examples
    )

    # Check all targets exist
    for f in "${check_targets[@]}"; do test -e "$f"; done

    flake8 --version
    flake8 "${check_targets[@]}"

    autopep8 --version
    autopep8 "${check_targets[@]}" -r --diff --exit-code

    # Detect invalid escape sequences in docstrings.
    # To workaround Travis issue (https://github.com/travis-ci/travis-ci/issues/7261),
    # ignore DeprecationWarning raised in `site.py`.
    python -Werror::DeprecationWarning -Wignore::DeprecationWarning:site -m compileall -f -q "${check_targets[@]}"
}


step_chainer_install_from_sdist() {
    # Build sdist.
    # sdist does not support out-of-source build.
    pushd "$REPO_DIR"
    python setup.py sdist
    popd

    # Install from sdist
    local envs=(
        CHAINERX_BUILD_TYPE=Debug
    )
    if [ -z "$MAKEFLAGS" ] ; then
        envs+=(MAKEFLAGS=-j"$DEFAULT_JOBS")
    fi

    if [[ $SKIP_CHAINERX != 1 ]]; then
        envs+=(CHAINER_BUILD_CHAINERX=1)
    fi
    env "${envs[@]}" pip install "$REPO_DIR"/dist/*.tar.gz
}


step_chainer_tests() {
    local mark="not slow and not gpu and not cudnn and not ideep"

    # On Windows theano fails to import
    if [[ $TRAVIS_OS_NAME == "windows" ]]; then
        mark="$mark and not theano"
    fi

    pytest -rfEX -m "$mark" "$REPO_DIR"/tests/chainer_tests
}


step_chainerx_python_tests() {
    pytest -rfEX "$REPO_DIR"/tests/chainerx_tests
}


step_chainermn_tests() {
    for NP in 1 2; do
        OMP_NUM_THREADS=1 \
            mpiexec -n ${NP} pytest -s -v -m 'not gpu and not slow' "$REPO_DIR"/tests/chainermn_tests
    done
}


step_docs() {
    SPHINXOPTS=-W make -C "$REPO_DIR"/docs html;
}


step_chainerx_cmake() {
    CHAINERX_BUILD_DIR="$WORK_DIR"/chainerx_build
    mkdir -p "$CHAINERX_BUILD_DIR"
    pushd "$CHAINERX_BUILD_DIR"

    cmake \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCHAINERX_BUILD_CUDA=OFF \
        -DCHAINERX_BUILD_TEST=ON \
        -DCHAINERX_BUILD_PYTHON=OFF \
        -DCHAINERX_WARNINGS_AS_ERRORS=ON \
        -DCMAKE_INSTALL_PREFIX="$WORK_DIR"/install_target \
        "$REPO_DIR"/chainerx_cc
    popd

    echo "CHAINERX_BUILD_DIR=\"$CHAINERX_BUILD_DIR\"" >> "$CHAINER_BASH_ENV"
}


step_chainerx_make() {
    make -C "$CHAINERX_BUILD_DIR" --output-sync
}


step_chainerx_ctest() {
    pushd "$CHAINERX_BUILD_DIR"
    ctest -V
    popd
}


step_before_install_chainerx_style_check_deps() {
    [ $TRAVIS_OS_NAME = "linux" ]  # currently only tested in linux

    sudo apt-get install -y \
         clang-format-6.0 \
         parallel \

}


step_install_chainerx_style_check_deps() {
    pip install cpplint
}


step_chainerx_cpplint() {
    "$REPO_DIR"/chainerx_cc/scripts/run-cpplint.sh --jobs "$DEFAULT_JOBS"
}


step_chainerx_clang_format() {
    "$REPO_DIR"/chainerx_cc/scripts/run-clang-format.sh --jobs "$DEFAULT_JOBS"
}


step_chainerx_cmake() {
    CHAINERX_BUILD_DIR="$WORK_DIR"/chainerx_build
    mkdir -p "$CHAINERX_BUILD_DIR"
    pushd "$CHAINERX_BUILD_DIR"

    cmake \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCHAINERX_BUILD_CUDA=OFF \
        -DCHAINERX_BUILD_TEST=ON \
        -DCHAINERX_BUILD_PYTHON=OFF \
        -DCHAINERX_WARNINGS_AS_ERRORS=ON \
        -DCMAKE_INSTALL_PREFIX="$WORK_DIR"/install_target \
        "$REPO_DIR"/chainerx_cc
    popd

    echo "CHAINERX_BUILD_DIR=\"$CHAINERX_BUILD_DIR\"" >> "$CHAINER_BASH_ENV"
}


step_chainerx_clang_tidy() {
    local target="$1"  # normal or test

    pushd "$CHAINERX_BUILD_DIR"
    "$REPO_DIR"/chainerx_cc/scripts/run-clang-tidy.sh "$target"
    popd
}
