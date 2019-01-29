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


# Assign default values
: "${MATRIX_EVAL:=}"
: "${SKIP_CHAINERX:=0}"
: "${CHAINER_TEST_STATUS:=0}"

REPO_DIR="$TRAVIS_BUILD_DIR"
WORK_DIR="$TRAVIS_BUILD_DIR"/_workspace
mkdir -p "$WORK_DIR"

# Env script which is sourced before each step
CHAINER_BASH_ENV="$WORK_DIR"/_chainer_bash_env
touch "$CHAINER_BASH_ENV"

export REPO_DIR
export WORK_DIR
export CHAINER_BASH_ENV


run_prestep() {
    # Failure immediately stops the script.
    bash "$this_dir"/run-step.sh "$@"
}


run_step() {
    # In case of failure, CHAINER_TEST_STATUS is incremented by 1.
    bash "$this_dir"/run-step.sh "$@" || CHAINER_TEST_STATUS=$((CHAINER_TEST_STATUS + 1))
}


case "${CHAINER_TRAVIS_TEST}" in
    "python-static-check")
        case "$phase" in
            before_install)
            ;;
            install)
                run_prestep install_chainer_style_check_deps
                run_prestep python_mypy_check_deps
            ;;
            script)
                run_step python_style_check
                run_step python_mypy_check
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
                run_prestep before_install_chainermn_test_deps
                ;;

            install)
                pip install -U pip wheel

                run_prestep install_chainer_test_deps
                run_prestep install_chainer_docs_deps
                run_prestep install_chainermn_test_deps

                run_prestep chainer_install_from_sdist
                ;;

            script)
                run_step chainer_tests
                run_step chainermn_tests

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
    *)
        echo "Unknown value of CHAINER_TRAVIS_TEST: ${CHAINER_TRAVIS_TEST}" >&2
        exit 1
        ;;
esac

# In "script" phases, the number of failed steps is assigned to this variable.
exit $CHAINER_TEST_STATUS
