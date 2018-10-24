# Multi-Layer Perceptron for MNIST Classification

This is a minimal example to write a feed-forward net.
The code consists of three parts: dataset preparation, network and optimizer definition and learning loop.
This is a common routine to write a learning process of networks with dataset that is small enough to fit into memory.

If you want to run this example on the N-th GPU, pass `--gpu=N` to the script.

## Training

There are 4 training scripts for the same model.

### `train_mnist.py`

This is the simplest example to train a simple multi-layer perceptron model using `Trainer` class with single GPU or CPUs.

Usage (with a GPU):
```
./train_mnist.py -g 0
```

Usage (with CPUs):
```
./train_mnist.py
```

Then the following files will be created after the training script finishes:

```
-- result
   |-- cg.dot
   |-- log
   `-- snapshot_iter_12000
```

`cg.dot` is a Graphviz dot file that contains the graph structure of the neural network used in this training script.
`log` is a JSON file that contains information during training.
`snapshot_iter_12000` is a snapshot of `Trainer` object which enables to resume the training.
You can also extract the trained model parameters from this snapshot file.

If you have installed [matplotlib](https://matplotlib.org/), the following files will be saved in the `result` directory:

```
accuracy.png
loss.png
```

These images display the changes in accuracy and loss values during training.

### `train_mnist_custom_loop.py`


This example shows how you can write the training loop by yourself without using the `Trainer` class abstraction.

Usage (with a GPU):
```
./train_mnist_custom_loop.py -g 0
```

Usage (with CPUs):
```
./train_mnist_custom_loop.py
```

Then, the following files will be created:

```
-- result
   |-- mlp.model
   `-- mlp.state
```

`mlp.model` contains the trained parameters of the model.
`mlp.state` contains the internal states of the Adam optimizer after the training.

### `train_mnist_data_parallel.py`

This example shows how to use multiple GPUs with the `Trainer` class in a data-parallel manner.
This script does not support CPUs and **requires at least two GPUs.**

Usage:
```
./train_mnist_data_parallel.py
```

Then, the following files will be created:

```
-- result_data_parallel
   |-- cg.dot
   |-- log
   `-- snapshot_iter_3000
```

### `train_mnist_model_parallel.py`

This example shows how to split a single model to different GPUs and train it with the `Trainer` class.
This script does not support CPUs and **requires at least two GPUs.**

Usage:
```
./train_mnist_model_parallel.py
```

Then, the following files will be created:

```
-- result_model_parallel
   |-- cg.dot
   |-- log
   `-- snapshot_iter_12000
```

## Inference

`inference.py` shows how to load the saved parameters trained by one of the above training scripts and use it for making a prediction on a test datum. It requires a saved snapshot file containing learned parameters, so please run one of the above training script first.

Usage (when you used `train_mnist.py`):
```
./inference.py --snapshot result/snapshot_iter_12000
```

Usage (when you used `train_mnist_custom_loop.py`):
```
./inference.py --snapshot result/mlp.model
```

Usage (when you used `train_mnist_data_parallel.py`):
```
./inference.py --snapshot result_data_parallel/snapshot_iter_3000
```

Usage (when you used `train_mnist_model_parallel.py`):
```
./inference.py --snapshot result_model_parallel/snapshot_iter_12000
```

You can run the inference on GPU by adding `--gpu 0` option to all the above commands.
