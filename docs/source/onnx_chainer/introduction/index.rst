.. _introduction:

Introduction
============

ONNX-Chainer converts Chainer model to ONNX format, export it.

Installation
------------

Install dependencies using ``pip`` via PyPI::

  $ pip install 'onnx<1.7.0'


Quick Start
-----------

First, install `ChainerCV <https://github.com/chainer/chainercv>`__ to get the pre-trained models.

.. code-block:: python

  import numpy as np

  import chainer
  import chainercv.links as C
  import onnx_chainer

  model = C.VGG16(pretrained_model='imagenet')

  # Pseudo input
  x = np.zeros((1, 3, 224, 224), dtype=np.float32)

  onnx_chainer.export(model, x, filename='vgg16.onnx')

``vgg16.onnx`` file will be exported.

Other export examples are put on `onnx_chainer/examples <https://github.com/chainer/chainer/tree/master/onnx_chainer/examples>`__. Please check them.

Supported Functions
-------------------

Currently 82 Chainer Functions are supported to export in ONNX format.

**Activation**

* ClippedReLU
* ELU
* HardSigmoid
* LeakyReLU
* LogSoftmax
* PReLUFunction
* ReLU
* Sigmoid
* Softmax
* Softplus
* Tanh

**Array**

* Cast
* Concat
* Copy
* Depth2Space
* Dstack
* ExpandDims
* GetItem
* Hstack
* Pad [#pad1]_ [#pad2]_
* Permutate
* Repeat
* Reshape
* ResizeImages
* Separate
* Shape [#shape1]_
* Space2Depth
* SplitAxis
* Squeeze
* Stack
* Swapaxes
* Tile
* Transpose
* Vstack
* Where

**Connection**

* Convolution2DFunction
* ConvolutionND
* Deconvolution2DFunction
* DeconvolutionND
* EmbedIDFunction [#embed1]_
* LinearFunction

**Loss**

* SoftmaxCrossEntropy

**Math**

* Absolute
* Add
* AddConstant
* ArgMax
* ArgMin
* BroadcastTo
* Clip
* Div
* DivFromConstant
* Exp
* Identity
* LinearInterpolate
* LogSumExp
* MatMul
* Max
* Maximum
* Mean
* Min
* Minimum
* Mul
* MulConstant
* Neg
* PowConstVar
* PowVarConst
* PowVarVar
* Prod
* RsqrtGPU
* Sqrt
* Square
* Sub
* SubFromConstant
* Sum

**Noise**

* Dropout [#dropout1]_

**Normalization**

* BatchNormalization
* FixedBatchNormalization
* LocalResponseNormalization
* NormalizeL2

**Pooling**

* AveragePooling2D
* AveragePoolingND
* MaxPooling2D
* MaxPoolingND
* ROIPooling2D
* Unpooling2D



.. [#pad1] mode should be either 'constant', 'reflect', or 'edge'
.. [#pad2] ONNX doesn't support multiple constant values for Pad operation
.. [#embed1] Current ONNX doesn't support ignore_label for EmbedID
.. [#dropout1] In test mode, all dropout layers aren't included in the exported file
.. [#shape1] Chainer doesn't support Shape function


Tested Environments
-------------------

* OS

    * Ubuntu 16.04, 18.04
    * Windows 10

* Python 3.5.5, 3.6.7, 3.7.2
* ONNX 1.4.1, 1.5.0, 1.6.0

    * opset version 7, 8, 9, 10, 11

* ONNX-Runtime 0.5.0


Run Test
--------

1. Install test modules
~~~~~~~~~~~~~~~~~~~~~~~

First, test modules for testing::

  $ pip install -e .[test]
  $ pip install onnxruntime

Test on GPU environment requires Cupy::

  $ pip install cupy  # or cupy-cudaXX is useful


2. Run tests
~~~~~~~~~~~~

Next, run ``pytest``::

  $ pytest -m "not gpu" tests/onnx_chainer_tests

on GPU environment::

  $ pytest tests/onnx_chainer_tests


Contribution
------------

Any contribution to ONNX-Chainer is welcome!

* Python codes follow `Chainer Coding Guidelines <https://docs.chainer.org/en/stable/contribution.html#coding-guidelines>`__
