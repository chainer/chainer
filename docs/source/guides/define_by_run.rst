.. _define_by_run:

Define-by-Run
-------------

.. currentmodule:: chainer

As mentioned on the top page, Chainer is a flexible framework for neural networks.
One major goal is flexibility, so it must enable us to write complex architectures simply and intuitively.

Most existing deep learning frameworks are based on the **"Define-and-Run"** scheme.
That is, first a network is defined and fixed, and then the user periodically feeds it with mini-batches of training data.
Since the network is statically defined before any forward/backward computation, all the logic must be embedded into the network architecture as *data*.
Consequently, defining a network architecture in such systems (e.g. Caffe) follows a declarative approach.
Note that one can still produce such a static network definition using imperative languages (e.g. torch.nn, Theano-based frameworks, and TensorFlow).

In contrast, Chainer adopts a **"Define-by-Run"** scheme, i.e., the network is defined dynamically via the actual forward computation.
More precisely, Chainer stores the history of computation instead of programming logic.
This strategy enables us to fully leverage the power of programming logic in Python.
For example, Chainer does not need any magic to introduce conditionals and loops into the network definitions.
The Define-by-Run scheme is the core concept of Chainer.
We will show in this tutorial how to define networks dynamically.

This strategy also makes it easy to write multi-GPU parallelization, since logic comes closer to network manipulation.
We will review such amenities in later sections of this tutorial.
