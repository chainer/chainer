Step 2: Datasets and Evaluators
===============================

Following from the previous step, we continue to
explain general steps to modify your code for ChainerMN
through the MNIST example.
All of the steps below are optional,
although useful for many cases.


Scattering Datasets
~~~~~~~~~~~~~~~~~~~

If you want to keep the definition of 'one epoch' correct,
we need to scatter the dataset to all workers.

For this purpose, ChainerMN provides a method ``scatter_dataset``.
It scatters the dataset of worker 0 (i.e., the worker whose ``comm.rank`` is 0)
to all workers. The given dataset of other workers are ignored.
The dataset is split into sub datasets of almost equal sizes and scattered
to the workers. To create a sub dataset, ``chainer.datasets.SubDataset`` is
used.

The following line of code from the original MNIST example loads the dataset::

  train, test = chainer.datasets.get_mnist()


We modify it as follows. Only worker 0 loads the dataset, and then it is scattered to all the workers::

  if comm.rank == 0:
      train, test = chainer.datasets.get_mnist()
  else:
      train, test = None, None

  train = chainermn.scatter_dataset(train, comm)
  test = chainermn.scatter_dataset(test, comm)


Creating A Multi-Node Evaluator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This step is also an optional step, but useful when validation is
taking a considerable amount of time.
In this case, you can also parallelize the validation by using *multi-node evaluators*.

Similarly to multi-node optimizers, you can create a multi-node evaluator
from a standard evaluator by using method ``create_multi_node_evaluator``.
It behaves exactly the same as the given original evaluator
except that it reports the average of results over all workers.

The following line from the original MNIST example adds an evaluator extension to the trainer::
  trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

To create and use a multi-node evaluator, we modify that part as follows::

  evaluator = extensions.Evaluator(test_iter, model, device=device)
  evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
  trainer.extend(evaluator)


Suppressing Unnecessary Extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some of extensions should be invoked only by one of the workers.
For example, if the ``PrintReport`` extension is invoked by all of the workers,
many redundant lines will appear in your console.
Therefore, it is convenient to register these extensions
only at workers of rank zero as follows::

  if comm.rank == 0:
      trainer.extend(extensions.dump_graph('main/loss'))
      trainer.extend(extensions.LogReport())
      trainer.extend(extensions.PrintReport(
          ['epoch', 'main/loss', 'validation/main/loss',
           'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
      trainer.extend(extensions.ProgressBar())
