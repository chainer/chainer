#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.links as L
import numpy as np
from onnx_caffe2.backend import Caffe2Backend
from onnx_caffe2.backend import run_model
from onnx_caffe2.helper import save_caffe2_net

import onnx_chainer

# Instantiate a Chainer model (Chain object)
model = L.VGG16Layers()

# Prepare a dummy input
x = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Do not forget setting train flag off!
chainer.config.train = False

# Export to ONNX model
onnx_model = onnx_chainer.export(model, x)

# Get an output of Chainer model
y = model(x)
if isinstance(y, dict):
    y = y['prob']
chainer_out = y.array

# Convert ONNX model to Caffe2 model
init_net, predict_net = Caffe2Backend.onnx_graph_to_caffe2_net(
    onnx_model.graph, device='CPU')

# Save the Caffe2 model to disk
init_file = "./vgg16_init.pb"
predict_file = "./vgg16_predict.pb"
save_caffe2_net(init_net, init_file, output_txt=False)
save_caffe2_net(predict_net, predict_file, output_txt=True)

# Get an output of Caffe2 model
caffe2_out = run_model(onnx_model, [x])[0]

# Check those two outputs have same values
np.testing.assert_almost_equal(
    chainer_out, caffe2_out, decimal=5)
