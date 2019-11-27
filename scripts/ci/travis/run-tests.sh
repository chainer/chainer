#!/usr/bin/env bash
# This script defines the matrices in Travis CI.
set -eux

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

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

export CHAINER_CI=travis

# Assign default values
: "${MATRIX_EVAL:=}"
: "${SKIP_CHAINERX:=0}"
: "${SKIP_CHAINERMN:=0}"
: "${CHAINER_TEST_STATUS:=0}"

REPO_DIR="$TRAVIS_BUILD_DIR"
WORK_DIR="$TRAVIS_BUILD_DIR"/_workspace
mkdir -p "$WORK_DIR"

# Env script which is sourced before each step
CHAINER_BASH_ENV="$WORK_DIR"/_chainer_bash_env
touch "$CHAINER_BASH_ENV"
source "$CHAINER_BASH_ENV"

export REPO_DIR
export WORK_DIR
export CHAINER_BASH_ENV

# Increase the default columns for better browsability
export COLUMNS=120


run_prestep() {
    # Failure immediately stops the script.
    bash "$this_dir"/run-step.sh "$@"
}


run_step() {
    local status=0

    travis_fold_start "${1}" "$@"
    bash "$this_dir"/run-step.sh "$@" || status=$?
    travis_fold_end "${1}"

    # In case of failure, CHAINER_TEST_STATUS is incremented by 1.
    if [[ ${status} != 0 ]]; then
        CHAINER_TEST_STATUS=$((CHAINER_TEST_STATUS + 1))
        console_warning "Error: the above step failed with status ${status}"
    fi
}

travis_fold_start() {
    echo -e "travis_fold:start:$1\033[33;1m$2\033[0m"
}

travis_fold_end() {
    echo -e "\ntravis_fold:end:$1\r"
}

console_warning() {
    echo -e "\033[31;1;4m$@\033[0m"
}

case "${CHAINER_TRAVIS_TEST}" in
    "python-static-check")
        case "$phase" in
            before_install)
            ;;
            install)
                run_prestep install_chainer_style_check_deps
            ;;
            script)
                run_step python_style_check
            ;;
        esac
        ;;

    "c-static-check")
        case "$phase" in
            before_install)
                run_prestep before_install_chainerx_style_check_deps
            ;;
            install)
                run_prestep install_chainerx_style_check_deps
                run_prestep chainerx_cmake  # cmake is required for clang-tidy
            ;;
            script)
                run_step chainerx_cpplint
                run_step chainerx_clang_format

                run_step chainerx_clang_tidy normal
                run_step chainerx_clang_tidy test
            ;;
        esac
        ;;

    "chainer")
        case "$phase" in
            before_install)
                eval "${MATRIX_EVAL}"
                run_prestep before_install_chainer_test

                if [[ $SKIP_CHAINERMN != 1 ]]; then
                    run_prestep before_install_chainermn_test_deps
                fi

                if [[ $SKIP_CHAINERX != 1 ]]; then
                    run_prestep before_install_chainerx_test_deps
                fi

                if [[ $TRAVIS_OS_NAME == "windows" ]]; then
                    choco install python3

                    export PATH="/c/Python37:/c/Python37/Scripts:$PATH"
                    echo 'export PATH="/c/Python37:/c/Python37/Scripts:$PATH"' >> $CHAINER_BASH_ENV

                    python -m pip install -U pip
                fi
                ;;

            install)
                pip install -U pip wheel

                run_prestep install_chainer_test_deps
                run_prestep install_chainer_docs_deps

                if [[ $SKIP_CHAINERMN != 1 ]]; then
                    run_prestep install_chainermn_test_deps
                fi

                run_prestep chainer_install_from_sdist
                ;;

            script)
                run_step chainer_tests

                if [[ $SKIP_CHAINERMN != 1 ]]; then
                    run_step chainermn_tests
                fi

                if [[ $SKIP_CHAINERX != 1 ]]; then
                    CHAINERX_TEST_CUDA_DEVICE_LIMIT=0 \
                        run_step chainerx_python_tests
                fi

                if [[ $SKIP_CHAINERX != 1 ]]; then
                    CHAINER_DOCS_SKIP_LINKCODE=1 \
                        run_step docs
                else
                    echo "Documentation build is skipped as ChainerX is not available.";
                fi
                ;;
        esac
        ;;

    "examples")
        case "$phase" in
            before_install)
                run_prestep before_install_chainer_test
                ;;

            install)
                run_prestep install_chainer_test_deps
                run_prestep chainer_install_from_sdist
                run_prestep install_chainer_example_tests
                ;;

            script)
                run_step chainer_example_tests
                ;;
        esac
        ;;

    "chainerx-cpp")
        case "$phase" in
            before_install)
            ;;
            install)
                run_prestep chainerx_cmake
                run_prestep chainerx_make
            ;;
            script)
                run_step chainerx_ctest
            ;;
        esac
        ;;

    *)
        echo "Unknown value of CHAINER_TRAVIS_TEST: ${CHAINER_TRAVIS_TEST}" >&2
        exit 1
        ;;
esac

# In "script" phases, the number of failed steps is assigned to this variable.
exit $CHAINER_TEST_STATUS
