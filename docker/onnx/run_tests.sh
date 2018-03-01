#!/bin/bash

docker run --rm -v $PWD/../:/root/onnx-chainer \
mitmul/onnx-chainer:latest \
bash -c "cd /root/onnx-chainer && pip install -e . && LD_LIBRARY_PATH=/usr/local/lib py.test -vvvs tests"