# This script defines the test steps, used from run-step.sh.
# In each step, the entire script is sourced first and subsequently one of the step functions is invoked.
# Therefore, the global part of the script, outside the function defnitions, is run before every single steps.


# Test variables defined
test ! -z ${WORK_DIR:+x}
test ! -z ${REPO_DIR:+x}

export CONDA_DIR=${CONDA_DIR:-"$WORK_DIR"/conda}
export CHAINERX_DIR=${CHAINERX_DIR:-"$REPO_DIR"/chainerx_cc}
export DOWNLOAD_DIR=${DOWNLOAD_DIR:-"$WORK_DIR"/downloads}

mkdir -p "$WORK_DIR"


step_setup() {
    local this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    source "$this_dir"/chainerx/setup-ubuntu.sh
}


step_setup_conda() {
    local this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    source "$this_dir"/chainerx/setup-conda.sh "$DOWNLOAD_DIR"/conda "$CONDA_DIR"

    echo 'PATH="$CONDA_DIR"/bin:"$PATH"' >> "$CHAINERX_CI_BASH_ENV"
}


step_show_environment_info() {
    source activate testenv

    echo "whoami: $(whoami) ($(id -u $(whoami)))"
    echo "PWD: $PWD"

    # g++
    g++ --version

    # cmake
    cmake --version

    # python
    which python
    python --version

    # conda
    conda info -a

    # CUDA driver
    if [ -f /proc/driver/nvidia/version ]; then
        cat /proc/driver/nvidia/version
    fi

    # CUDA runtime
    if [ -f /usr/local/cuda/version.txt ]; then
        cat /usr/local/cuda/version.txt
    fi
}


step_setup_conda_environment() {
    source activate testenv

    reqs=(
        # hacking alternatives
        autopep8
        'pycodestyle<2.4.0,>=2.3'
        'pbr>=1.8'
        'pep8==1.5.7'
        'pyflakes<1.7.0,>=1.6.0'
        'flake8==3.5.0'
        'mccabe==0.6.1'
        'six>=1.9.0'

        pytest pytest-cov coveralls
        cpplint
    )

    pip install -U "${reqs[@]}"
}


step_python_style_check() {
    source activate testenv

    check_targets=(
        "$REPO_DIR"/*.py
        "$REPO_DIR"/chainer
        "$REPO_DIR"/chainermn
        "$REPO_DIR"/chainerx
        "$REPO_DIR"/tests
        "$REPO_DIR"/examples
        "$CHAINERX_DIR"/examples
    )

    # Check all targets exist
    for f in "${check_targets[@]}"; do test -e "$f"; done

    flake8 --version
    flake8 "${check_targets[@]}"

    autopep8 --version
    autopep8 "${check_targets[@]}" -r --diff | tee "$WORK_DIR"/check_autopep8
    test ! -s "$WORK_DIR"/check_autopep8
}


step_clang_format() {
    "$CHAINERX_DIR"/scripts/run-clang-format.sh --jobs 4
}


step_cpplint() {
    source activate testenv

    "$CHAINERX_DIR"/scripts/run-cpplint.sh --jobs 4
}


step_setup_openblas() {
    # Install openblas on another directory with conda testenv because, otherwise, we get warnings like
    # /usr/bin/cmake: /root/miniconda/envs/testenv/lib/libssl.so.1.0.0: no version information available (required by /usr/lib/x86_64-linux-gnu/libcurl.so.4)

    conda create -y -q --name openblasenv openblas
    source activate openblasenv

    echo "export LD_LIBRARY_PATH=\"$CONDA_PREFIX/lib:$LD_LIBRARY_PATH\"" >> $CHAINERX_CI_BASH_ENV
    echo "export CPATH=\"$CONDA_PREFIX/include:$CPATH\"" >> $CHAINERX_CI_BASH_ENV
}


step_cmake() {
    mkdir -p "$WORK_DIR"/build
    pushd "$WORK_DIR"/build

    # -DPYTHON_EXECUTABLE:FILEPATH is specified in order to use the created environment when building pybind11 instead of the default Python in miniconda
    #CUDNN_ROOT_DIR=$HOME/.cudnn/active \
    cmake \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCHAINERX_BUILD_CUDA=ON \
        -DCHAINERX_BUILD_TEST=ON \
        -DCHAINERX_BUILD_PYTHON=OFF \
        -DCHAINERX_WARNINGS_AS_ERRORS=ON \
        -DCMAKE_INSTALL_PREFIX="$WORK_DIR"/install_target \
        "$CHAINERX_DIR"
    popd
}


step_clang_tidy() {
    source activate testenv

    local target="$1"  # normal or test

    pushd "$WORK_DIR"/build
    "$CHAINERX_DIR"/scripts/run-clang-tidy.sh "$target"
    popd
}


step_make() {
    make -C "$WORK_DIR"/build --output-sync
}


step_make_install() {
    make -C "$WORK_DIR"/build install
}



step_ctest() {
    pushd "$WORK_DIR"/build
    ctest -V
    popd
}


step_python_build() {
    source activate testenv

    CHAINER_BUILD_CHAINERX=1 \
    CHAINERX_BUILD_CUDA=ON \
    pip install "$REPO_DIR"[test]
}


step_python_test_chainerx() {
    source activate testenv

    # TODO(niboshi): threshold is temporarily lowered from 80 to 50. Restore it after writing tests for testing package.
    COVERAGE_FILE="$WORK_DIR"/coverage-data \
    pytest \
        --showlocals \
        --cov=chainerx \
        --no-cov-on-fail \
        --cov-fail-under=50 \
        --cov-report html:"$WORK_DIR"/coverage-html/python \
        "$REPO_DIR"/tests/chainerx_tests
}


step_python_test_chainer() {
    source activate testenv

    # Some chainer tests generate files under current directory.
    local temp_dir="$(mktemp -d)"
    pushd "$temp_dir"

    pytest \
        --showlocals \
        -m 'not slow and not ideep' \
        "$REPO_DIR"/tests/chainer_tests

    popd
}
