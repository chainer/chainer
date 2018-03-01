#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import namedtuple

import chainer
import chainer.links as L
import numpy as np

import mxnet as mx
import onnx_chainer
import onnx_mxnet

chainer.config.train = False


def save_as_onnx_then_import_from_mxnet(model, fn):
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    chainer_out = model(x)['prob'].array

    onnx_chainer.export(model, x, fn)

    sym, params = onnx_mxnet.import_model(fn)

    mod = mx.mod.Module(
        symbol=sym, data_names=['input_0'], context=mx.cpu(), label_names=None)
    mod.bind(
        for_training=False, data_shapes=[('input_0', x.shape)],
        label_shapes=None)
    mod.set_params(arg_params=params, aux_params=None, allow_missing=True)

    Batch = namedtuple('Batch', ['data'])
    mod.forward(Batch([mx.nd.array(x)]))

    mxnet_out = mod.get_outputs()[0].asnumpy()

    print(mxnet_out.shape)

    np.testing.assert_almost_equal(
        chainer_out, mxnet_out, decimal=5)


def main():
    model = L.VGG16Layers()
    save_as_onnx_then_import_from_mxnet(model, 'vgg16.onnx')

    model = L.ResNet50Layers()
    save_as_onnx_then_import_from_mxnet(model, 'resnet50.onnx')


if __name__ == '__main__':
    main()
