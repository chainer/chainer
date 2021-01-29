# HYPERPARAMETER OPTIMIZATION WITH OPTUNA 

## Requirements

- Optuna

## Description

[chainer_simple.py](https://github.com/optuna/optuna/blob/master/examples/chainer_simple.py)
Optuna example that optimizes multi-layer perceptrons using Chainer.
In this example, we optimize the validation accuracy of hand-written digit recognition using
Chainer and MNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole MNIST dataset, we here use a small
subset of it.

[chainermn_simple.py](https://github.com/optuna/optuna/blob/master/examples/chainermn_simple.py)
Optuna example that optimizes multi-layer perceptrons using ChainerMN.
In this example, we optimize the validation accuracy of hand-written digit recognition using
ChainerMN and MNIST, where architecture of neural network is optimized.

chainer_integration.py
Optuna example that demonstrates a pruner for Chainer.
In this example, we optimize the hyperparameters of a neural network for hand-written
digit recognition in terms of validation loss. The network is implemented by Chainer and
evaluated by MNIST dataset. Throughout the training of neural networks, a pruner observes
intermediate results and stops unpromising trials.

[chainermn_integration.py](https://github.com/optuna/optuna/blob/master/examples/pruning/chainermn_integration.py)
Optuna example that demonstrates a pruner for ChainerMN.
In this example, we optimize the validation accuracy of hand-written digit recognition using
ChainerMN and MNIST, where architecture of neural network is optimized. Throughout the training of
neural networks, a pruner observes intermediate results and stops unpromising trials.
