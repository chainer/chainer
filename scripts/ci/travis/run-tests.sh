#!/usr/bin/env bash
# This script defines CI steps to be run in Travis CI.
# TODO(niboshi): Definitions of the steps could be merged with scripts/ci/steps.sh.

set -eux


phase="$1"

# Check phase argument
case "$phase" in
    before_install|install|script)
        ;;
    *)
        echo "Unknown phase: ${phase}" >&2
        exit 1
        ;;
esac


# Assign default values
: "${MATRIX_EVAL:=}"
: "${SKIP_CHAINERX:=0}"

DEFAULT_JOBS=2
REPO_DIR="$TRAVIS_BUILD_DIR"
WORK_DIR="$TRAVIS_BUILD_DIR"/_workspace
mkdir -p "$WORK_DIR"


# Test step definitions


_install_chainer_deps() {
    # Install extras_require of Chainer.
    local extras="$1"

    # It's not possible to install only requirements.
    # Chainer is uninstalled after the installation.
    # TODO(niboshi): Use other installation tool
    pip install -e "$REPO_DIR"["$extras"]
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

    if python -c "import sys; assert sys.version_info >= (3, 4)"; then
        pip install -U 'mypy>=0.650';
    fi
}


step_before_install_chainer_test() {
    # Remove oclint as it conflicts with GCC (indirect dependency of hdf5)
    if [[ $TRAVIS_OS_NAME = "osx" ]]; then
        brew update >/dev/null
        brew outdated pyenv || brew upgrade --quiet pyenv

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
    local envs=(MAKEFLAGS=-j"$DEFAULT_JOBS")

    if [[ $SKIP_CHAINERX != 1 ]]; then
        envs+=(CHAINER_BUILD_CHAINERX=1)
    fi
    env "${envs[@]}" pip install "$REPO_DIR"/dist/*.tar.gz
}


step_chainer_tests() {
    pytest -m "not slow and not gpu and not cudnn and not ideep" "$REPO_DIR"/tests/chainer_tests
    if python -c "import sys; assert sys.version_info >= (3, 4)"; then
        (cd "$REPO_DIR" && mypy chainer)
    fi
}


step_chainermn_tests() {
    for NP in 1 2; do
        OMP_NUM_THREADS=1 \
            mpiexec -n ${NP} pytest -s -v -m 'not gpu and not slow' "$REPO_DIR"/tests/chainermn_tests
    done
}


step_docs() {
    make -C "$REPO_DIR"/docs html;
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
}


step_chainerx_clang_tidy() {
    local target="$1"  # normal or test

    pushd "$CHAINERX_BUILD_DIR"
    "$REPO_DIR"/chainerx_cc/scripts/run-clang-tidy.sh "$target"
    popd
}


run_step() {
    step="$1"
    shift
    echo "=== Step: $step $@"

    step_"$step" "$@"
}


case "${CHAINER_TRAVIS_TEST}" in
    "python-stylecheck")
        case "$phase" in
            before_install)
            ;;
            install)
                run_step install_chainer_style_check_deps
            ;;
            script)
                run_step python_style_check
            ;;
        esac
        ;;

    "c-stylecheck")
        case "$phase" in
            before_install)
                run_step before_install_chainerx_style_check_deps
            ;;
            install)
                run_step install_chainerx_style_check_deps
            ;;
            script)
                run_step chainerx_cpplint
                run_step chainerx_clang_format

                run_step chainerx_cmake  # cmake is required for clang-tidy
                run_step chainerx_clang_tidy normal
                run_step chainerx_clang_tidy test
            ;;
        esac
        ;;

    "chainer")
        case "$phase" in
            before_install)
                eval "${MATRIX_EVAL}"
                run_step before_install_chainer_test
                run_step before_install_chainermn_test_deps
                ;;

            install)
                pip install -U pip wheel

                run_step install_chainer_test_deps
                run_step install_chainer_docs_deps
                run_step install_chainermn_test_deps
                ;;

            script)
                run_step chainer_install_from_sdist
                run_step chainer_tests
                run_step chainermn_tests

                if [[ $SKIP_CHAINERX != 1 ]]; then
                    CHAINER_DOCS_SKIP_LINKCODE=1 \
                        run_step docs
                else
                    echo "Documentation build is skipped as ChainerX is not available.";
                fi
                ;;
        esac
        ;;
    *)
        echo "Unknown value of CHAINER_TRAVIS_TEST: ${CHAINER_TRAVIS_TEST}" >&2
        exit 1
        ;;
esac
