Trainer
~~~~~~~

When we want to train neural networks, we have to run *training loops* that update the parameters many times.
A typical training loop consists of the following procedures:

1. Iterations over training datasets
2. Preprocessing of extracted mini-batches
3. Forward/backward computations of the neural networks
4. Parameter updates
5. Evaluations of the current parameters on validation datasets
6. Logging and printing of the intermediate results

Chainer provides a simple yet powerful way to make it easy to write such training processes.
The training loop abstraction mainly consists of two components:

- **Dataset abstraction**.
  It implements 1 and 2 in the above list.
  The core components are defined in the :mod:`~chainer.dataset` module.
  There are also many implementations of datasets and iterators in :mod:`~chainer.datasets` and :mod:`~chainer.iterators` modules, respectively.
- **Trainer**.
  It implements 3, 4, 5, and 6 in the above list.
  The whole procedure is implemented by :class:`~training.Trainer`.
  The way to update parameters (3 and 4) is defined by :class:`~training.Updater`, which can be freely customized.
  5 and 6 are implemented by instances of :class:`~training.Extension`, which appends an extra procedure to the training loop.
  Users can freely customize the training procedure by adding extensions. Users can also implement their own extensions.

