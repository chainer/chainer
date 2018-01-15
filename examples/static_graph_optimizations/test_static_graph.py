#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Example of 'static graph' feature by Brian Vogel.
# Requires Chainer v3.
# Note: this code is proof-of-concept only. It does not yet
# implement optimized functions and only supports the
# Linear link/function.

import argparse

import chainer
from chainer.dataset import convert
import chainer.links as L
import numpy as np
from chainer.datasets import get_mnist
from chainer import optimizers
import chainer.links.model.classifier as classifier

from chainer.graph_optimizations.static_graph import static_graph

# todo (vogel)

# For now, please refer to the existing examples that have been optimizaed to use
# this feature:
# MNIST example, CIFAR example, ptb example.

if __name__ == '__main__':

    # todo: consider a modified ptb or char-rnn example in which the LSTM node is written
    # in define-by-run code (for gates, activations, linear layers etc). This
    # would be a good use case for static graph optimizations. For example, if a
    # researcher is experimenting with a novel type of RNN, it would be preferable
    # to be able to write a quick prototype implementation in terms of basic chainer
    # functions using define-by-run code and then simply enable static
    # graph optimizations to get improved runtime performance.
