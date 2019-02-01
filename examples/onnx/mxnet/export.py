#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections

import numpy as np

import chainer
import chainer.functions as F
import chainercv.links as C
import mxnet
import onnx_chainer


def save_as_onnx_then_import_from_mxnet(model, fn):
    # Prepare an input tensor
    x = np.random.rand(1, 3, 224, 224).astype(np.float32) * 255

    # Run the model on the data
    with chainer.using_config('train', False):
        chainer_out = model(x).array

    # Export Chainer model into ONNX
    onnx_chainer.export(model, x, fn)

    # Load ONNX model into MXNet symbol
    sym, arg, aux = mxnet.contrib.onnx.import_model(fn)

    # Find the name of input tensor
    data_names = [graph_input for graph_input in sym.list_inputs()
                  if graph_input not in arg and graph_input not in aux]
    data_shapes = [(data_names[0], x.shape)]

    # Create MXNet model
    mod = mxnet.mod.Module(
        symbol=sym, data_names=data_names, context=mxnet.cpu(),
        label_names=None)
    mod.bind(
        for_training=False, data_shapes=data_shapes,
        label_shapes=None)
    mod.set_params(
        arg_params=arg, aux_params=aux, allow_missing=True,
        allow_extra=True)

    # Create input data
    Batch = collections.namedtuple('Batch', ['data'])
    input_data = Batch([mxnet.nd.array(x)])

    # Forward computation using MXNet
    mod.forward(input_data)

    # Retrieve the output of forward result
    mxnet_out = mod.get_outputs()[0].asnumpy()

    # Check the prediction results are same
    assert np.argmax(chainer_out) == np.argmax(mxnet_out)

    # Check both outputs have same values
    np.testing.assert_almost_equal(chainer_out, mxnet_out, decimal=5)


def main():
    model = C.VGG16(pretrained_model='imagenet')
    save_as_onnx_then_import_from_mxnet(model, 'vgg16.onnx')

    model = C.ResNet50(pretrained_model='imagenet', arch='he')
    # Change cover_all option to False
    # to match the default behavior of MXNet's pooling
    model.pool1 = lambda x: F.max_pooling_2d(
        x, ksize=3, stride=2, cover_all=False)
    save_as_onnx_then_import_from_mxnet(model, 'resnet50.onnx')


if __name__ == '__main__':
    main()
