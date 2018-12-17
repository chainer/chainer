#!/usr/bin/env bash
set -eu

# Dump all the environment variables starting with CHAINERX_
for var in ${!CHAINERX_@}; do
    echo "$var=${!var}"
done


add_path() {
    local var="$1"
    local path="$2"
    local old="${!var:-}"

    test -d "$path"  # Check if the path exists

    if [ -z "$old" ]; then
        export $var="$path"
    else
        export $var="$path":"$old"
    fi
}

ensure_dir_exists() {
    local var="$1"
    local path="${!var}"
    if [ ! -d "$path" ]; then
        echo "Directory not found (as $var): $path" >&2
        exit 1
    fi
}

# Config variables
export WORK_DIR="$CHAINERX_JENKINS_WORK_DIR"
export REPO_DIR="$CHAINERX_JENKINS_REPO_DIR"
export CONDA_DIR="$CHAINERX_JENKINS_CONDA_DIR"


ensure_dir_exists WORK_DIR
ensure_dir_exists REPO_DIR
ensure_dir_exists CONDA_DIR

add_path PATH "$CONDA_DIR"/bin


run_step() {
    # Runs a single step
    bash "$REPO_DIR"/chainerx_cc/scripts/ci/run-step.sh "$@"
}


# Run steps

run_step show_environment_info

case "${CHAINERX_JENKINS_TEST_TYPE}" in
    'misc')
        run_step setup_conda_environment
        run_step python_style_check
        run_step clang_format
        run_step cpplint
        run_step cmake
        run_step clang_tidy normal
        run_step clang_tidy test
        ;;
    'chainerx-c')
        run_step setup_openblas
        run_step cmake
        CHAINERX_NVCC_GENERATE_CODE=arch=compute_50,code=sm_50 MAKEFLAGS=-j16 run_step make
        run_step make_install
        run_step ctest
        ;;
    'chainerx-py3')
        run_step setup_conda_environment
        CHAINERX_NVCC_GENERATE_CODE=arch=compute_50,code=sm_50 MAKEFLAGS=-j16 run_step python_build
        run_step python_test_chainerx
        ;;
    'chainer-py3')
        run_step setup_conda_environment
        CHAINERX_NVCC_GENERATE_CODE=arch=compute_50,code=sm_50 MAKEFLAGS=-j16 run_step python_build
        run_step python_test_chainer
        ;;
    *)
        echo "Unknown value of CHAINERX_JENKINS_TEST_TYPE: ${CHAINERX_JENKINS_TEST_TYPE}" >&2
        exit 1
        ;;
esac
