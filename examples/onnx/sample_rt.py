#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import onnxruntime as rt

import chainer
import chainer.links as L
import onnx_chainer
from chainercv import transforms
from PIL import Image

model = L.VGG16Layers()

# Load image
img = np.asarray(Image.open('images/cat.jpg'))

# Convert RGB to BGR
img = img[:, :, ::-1]

# Subtract the mean value of ImageNet train dataset (BGR)
img = img - np.array([[[103.939, 116.779, 123.68]]])

# Transpose the image array from (H, W, C) to (C, H, W)
img = img.transpose(2, 0, 1)

# Crop the center region
img = transforms.center_crop(img, (224, 224))

# Create a minibatch whose batchsize=1
x = np.asarray([img], dtype=np.float32)

# Export to ONNX
onnx_model = onnx_chainer.export(model, x)

# Create a ONNXRuntime session
sess = rt.InferenceSession(onnx_model.SerializeToString())

# Create a prediction result with ONNXRuntime
input_names = [i.name for i in sess.get_inputs()]
rt_out = sess.run(
    None, {name: array for name, array in zip(input_names, (x,))})

# Create a prediction result with Chainer
with chainer.using_config('train', False), \
        chainer.using_config('enable_backprop', False):
    chainer_out = model(x)['prob']

# Calculate the highest class predicted by each model
ch_pred = np.argmax(chainer_out.array[0])
rt_pred = np.argmax(rt_out[0])

id_name = [l.strip() for l in open('synset_words.txt').readlines()]

print('Prediction by Chainer:\t', id_name[ch_pred])
print('Prediction by ONNXRuntime:\t', id_name[rt_pred])
print('...should be same.')

np.testing.assert_almost_equal(rt_out[0], chainer_out.array, decimal=5)
