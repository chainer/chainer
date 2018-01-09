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
#from chainer import static_graph


#from chainer.graph_optimizations import static_graph
from chainer.graph_optimizations.static_graph import static_graph

# Network definition (borrowed from the existing Chainer MNIST example).
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1=L.Linear(None, n_units),  # n_in -> n_units
            l2=L.Linear(None, n_units),  # n_units -> n_units
            l3=L.Linear(None, n_out),  # n_units -> n_out
        )

    # To use the static graph feature, just add the `@static_graph' decorator to the
    # `__call__()` method of a Chain.
    @static_graph
    def __call__(self, x):
        # Only the Linear function currently supports the static graph feature.
        h1 = self.l1(x)
        h2 = self.l2(h1)
        return self.l3(h2)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Chainer example: static graph')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    model = classifier.Classifier(MLP(200, 10))
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    train, valid = get_mnist()
    train_count = len(train)
    batchsize = 50
    # We don't shuffle the training examples for easier testing.
    train_iter = chainer.iterators.SerialIterator(train, batchsize, shuffle=False)

    cur_iteration = 0
    while cur_iteration < 10:
        batch = train_iter.next()
        x_array, t_array = convert.concat_examples(batch, args.gpu)
        x = chainer.Variable(x_array)
        t = chainer.Variable(t_array)
        loss = model(x, t)

        # You MUST set retain_grad=True when calling loss.backward() on a model that
        # uses static_graph.
        print('Loss: ', loss.data)
        #todo: the backward() method does not currently accumuate gradients correctly for
        # general graphs. Need to modify Varible.backward() to make it work correctly.
        loss.backward(retain_grad=True)

        # todo: Optimizers do not yet support static graph feature.

        cur_iteration += 1




