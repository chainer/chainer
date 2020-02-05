#! /usr/bin/env sh
set -eux

export INSTALL_CUPY=""
export EXAMPLE_ARGS=""
export DOCKER_RUNTIME_ARG=""
export PYTEST_ARGS="-m \"not gpu\""
if [ -n "${GPU+x}" ]; then
    export INSTALL_CUPY="on"
    export EXAMPLE_ARGS="-G "${GPU}
    export DOCKER_RUNTIME_ARG="--runtime=nvidia"
    export PYTEST_ARGS=""
fi
if [ -z "${ONNX_VER+x}" ]; then export ONNX_VER=""; fi

cat <<EOM >test_script.sh
set -eux

if [[ "${INSTALL_CUPY}" == "on" ]]; then pip install --pre cupy-cuda101; fi
pip install -e .[test]
pip install 'onnx<1.7.0' onnxruntime
if [[ "${ONNX_VER}" != "" ]]; then pip install onnx==${ONNX_VER}; fi
pip install pytest-cov
pip list -v
pytest -x -s -vvvs ${PYTEST_ARGS} tests/onnx_chainer_tests --cov onnx_chainer

pip install chainercv
export CHAINERCV_DOWNLOAD_REPORT="OFF"
for dir in \`ls onnx_chainer/examples\`
do
  if [[ -f onnx_chainer/examples/\${dir}/export.py ]]; then
    python onnx_chainer/examples/\${dir}/export.py -T ${EXAMPLE_ARGS}
  fi
done
EOM

docker run ${DOCKER_RUNTIME_ARG} -i --rm \
    -v $(pwd):/root/chainer --workdir /root/chainer \
    disktnk/onnx-chainer:ci-py${PYTHON_VER} \
    /bin/bash /root/chainer/test_script.sh
