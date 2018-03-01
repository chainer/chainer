#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

import nnvm
import nnvm.compiler
import onnx_chainer


class MLP(chainer.Chain):

    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.l1 = L.Convolution2D(None, 10, 3, 1, 1)

    def __call__(self, x):
        return F.relu(self.l1(x))


# Currently it doesn't work

model = MLP()
x = np.random.rand(1, 3, 28, 28).astype(np.float32)

chainer.config.train = False

model_onnx = onnx_chainer.export(model, x)

sym, params = nnvm.frontend.from_onnx(model_onnx)
target = 'llvm'
shape_dict = {model_onnx.graph.input[-1].name: x.shape}
nnvm.compiler.build(sym, target, shape_dict, params=params, dtype="float32")
