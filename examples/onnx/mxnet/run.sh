#!/bin/bash

docker run \
-v $PWD/../../:/root/onnx-chainer \
-v $HOME/.chainer:/root/.chainer \
-ti mitmul/onnx-chainer:latest \
bash -c "cd /root/ && cd onnx-chainer && ls && python setup.py develop && /bin/bash"