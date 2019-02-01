#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import onnx

import chainer
import chainer.functions as F
import chainercv.links as C
import nnvm
import onnx_chainer
import tvm


def save_as_onnx_then_import_from_nnvm(model, fn):
    # Prepare an input tensor
    x = np.random.rand(1, 3, 224, 224).astype(np.float32) * 255

    # Run the model on the data
    with chainer.using_config('train', False):
        chainer_out = model(x).array

    # Export Chainer model into ONNX
    onnx_chainer.export(model, x, fn)

    # Load the saved ONNX file using ONNX module
    model_onnx = onnx.load(fn)

    # Convert the ONNX model object into NNVM symbol
    sym, params = nnvm.frontend.from_onnx(model_onnx)

    # Choose the compilation target
    target = 'llvm'

    # Extract the name of input variable in the ONNX graph
    input_name = sym.list_input_names()[0]
    shape_dict = {input_name: x.shape}

    # Compile the model using NNVM
    graph, lib, params = nnvm.compiler.build(
        sym, target, shape_dict, params=params,
        dtype={input_name: 'float32'})

    # Convert the compiled model into TVM module
    module = tvm.contrib.graph_runtime.create(graph, lib, tvm.cpu(0))

    # Set the input tensor x
    module.set_input(input_name, tvm.nd.array(x))
    module.set_input(**params)

    # Run the model
    module.run()

    # Retrieve the inference result
    out_shape = (1, 1000)
    output = tvm.nd.empty(out_shape, ctx=tvm.cpu(0))
    nnvm_output = module.get_output(0, output).asnumpy()

    # Check both outputs have same values
    np.testing.assert_almost_equal(chainer_out, nnvm_output, decimal=5)


def main():
    model = C.VGG16(pretrained_model='imagenet')
    save_as_onnx_then_import_from_nnvm(model, 'vgg16.onnx')

    model = C.ResNet50(pretrained_model='imagenet', arch='he')
    # Change cover_all option to False
    # to match the default behavior of MXNet's pooling
    model.pool1 = lambda x: F.max_pooling_2d(
        x, ksize=3, stride=2, cover_all=False)
    save_as_onnx_then_import_from_nnvm(model, 'resnet50.onnx')


if __name__ == '__main__':
    main()
