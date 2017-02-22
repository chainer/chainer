# Multi-Layer Perceptron for MNIST Classification

This is a minimal example to write a feed-forward net.
The code consists of three parts: dataset preparation, network and optimizer definition and learning loop.
This is a common routine to write a learning process of networks with dataset that is small enough to fit into memory.

If you want to run this example on the N-th GPU, pass `--gpu=N` to the script.

There are several versions of the training script:

- A data-parallel version: `train_mnist_data_parallel.py`

- A model-parallel version: `train_mnist_model_parallel.py`

- A version using trainer: `train_mnist_trainer.py`

- A version that does not use trainer: `train_mnist.py`

For begining users, we recommend to first look at `train_mnist.py` since this version uses a user-defined training loop in the training script. This allows the user to easily understand the operations of the training loop without having to first learn the trainer abstractions.

For more advanced users, we recommend to also look at `train_mnist_trainer.py`. This version uses Trainer, which is intended to reduce the amount of boilerplate code the user needs to write to define the training loop, including the training loop itself.

Whether or not to use trainer to define the training loop is a matter of personal preference and some models might be better suited for one or the other which is why we provide examples of both approaches.