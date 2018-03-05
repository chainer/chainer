
   In the example code of this tutorial, we assume for simplicity that the following symbols are already imported.

.. testcode::

     import numpy as np
     import chainer
     from chainer.backends import cuda
     from chainer import Function, gradient_check, report, training, utils, Variable
     from chainer import datasets, iterators, optimizers, serializers
     from chainer import Link, Chain, ChainList
     import chainer.functions as F
     import chainer.links as L
     from chainer.training import extensions

