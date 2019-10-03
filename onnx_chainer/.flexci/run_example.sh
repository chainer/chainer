#!/bin/bash

set -eux

export EXAMPLE_ARGS=$1
export CHAINERCV_DOWNLOAD_REPORT="OFF"

for dir in `ls onnx_chainer/examples`
do
  if [[ -f onnx_chainer/examples/${dir}/export.py ]]; then
    python onnx_chainer/examples/${dir}/export.py -T ${EXAMPLE_ARGS}
  fi
done
