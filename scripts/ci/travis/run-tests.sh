#!/usr/bin/env bash
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


case "${CHAINER_TRAVIS_TEST}" in
    "chainer")
        case "$phase" in
            before_install)
                eval "${MATRIX_EVAL}"
                # Remove oclint as it conflicts with GCC (indirect dependency of hdf5)
                if [[ $TRAVIS_OS_NAME = "osx" ]]; then
                    brew update >/dev/null;
                    brew outdated pyenv || brew upgrade --quiet pyenv;

                    PYTHON_CONFIGURE_OPTS="--enable-unicode=ucs2" pyenv install -ks $PYTHON_VERSION;
                    pyenv global $PYTHON_VERSION;
                    python --version;

                    brew cask uninstall oclint;
                    brew install hdf5;
                    brew install open-mpi;
                fi
                ;;

            install)
                pip install -U pip wheel
                if python -c "import sys; assert sys.version_info >= (3, 4)"; then
                    pip install -U 'mypy>=0.650';
                fi
                pip install mpi4py
                python setup.py sdist
                if [[ $SKIP_CHAINERX != 1 ]]; then
                    export CHAINER_BUILD_CHAINERX=1;
                fi
                MAKEFLAGS=-j2
                pip install dist/*.tar.gz
                MAKEFLAGS=-j2
                pip install -U -e .[travis]
                ;;

            script)
                flake8
                autopep8 -r . --diff --exit-code
                # To workaround Travis issue (https://github.com/travis-ci/travis-ci/issues/7261),
                # ignore DeprecationWarning raised in `site.py`.
                python -Werror::DeprecationWarning -Wignore::DeprecationWarning:site -m compileall -f -q chainer chainermn examples tests docs
                if python -c "import sys; assert sys.version_info >= (3, 4)"; then
                    mypy chainer;
                fi
                pushd tests
                pytest -m "not slow and not gpu and not cudnn and not ideep" chainer_tests
                export OMP_NUM_THREADS=1
                (for NP in 1 2; do mpiexec -n ${NP} pytest -s -v -m 'not gpu and not slow' chainermn_tests || exit $?; done)
                popd
                if [[ $TRAVIS_OS_NAME == "linux" ]]; then
                    python setup.py develop;
                fi
                if [[ $SKIP_CHAINERX != 1 ]]; then
                    make -C docs html;
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
