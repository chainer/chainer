# Multi-Layer Perceptron for MNIST Classification

This is a minimal example to write a feed-forward net.
The code consists of three parts: dataset preparation, network and optimizer definition and learning loop.
This is a common routine to write a learning process of networks with dataset that is small enough to fit into memory.

If you want to run this example on the N-th GPU, pass `--gpu=N` to the script.

## Training

There are 4 training scripts for the same model.

### `train_mnist.py`

This is the simplest example to train a simple multi-layer perceptron model using `Trainer` class with single GPU or CPUs.

### `train_mnist_custom_loop.py`


This example shows how you can write the training loop by yourself without using `Trainer` class abstraction.

### `train_mnist_data_parallel.py`

This example shows how to use multiple GPUs with `Trainer` class in data-parallel manner.

### `train_mnist_model_parallel.py`

This example shows how to split a single model to different GPUs and train it with `Trainer` class.

## Inference

`inference.py` shows how to load the saved parameters which is trained by one of the above training scripts and use it for making a prediction on a test datum. It requires a NPZ file containing learnt parameters, so please run one of the above training script first.
