.. module:: chainer.training


.. snapshot_writers is referred to from within chainer.training.snapshot docstring.
.. toctree::
   :hidden:

   snapshot_writers

Training Tools
=========================

Chainer provides a standard implementation of the training loops under the :mod:`chainer.training` module. It is built on top of many other core features of Chainer, including Variable and Function, Link/Chain/ChainList, Optimizer, Dataset, and Reporter/Summary. Compared to the training loop abstraction of other machine learning tool kits, Chainer's training framework aims at maximal flexibility, while keeps the simplicity for the typical usages. Most components are pluggable, and users can overwrite the definition.

The core of the training loop abstraction is :class:`~chainer.training.Trainer`, which implements the training loop itself. The training loop consists of two parts: one is :class:`~chainer.training.Updater`, which actually updates the parameters to train, and the other is :class:`~chainer.training.Extension` for arbitrary functionalities other than the parameter update.

Updater and some extensions use :mod:`chainer.dataset` and :class:`~chainer.dataset.Iterator` to scan the datasets and load mini-batches. The trainer also uses :class:`~chainer.Reporter` to collect the observed values, and some extensions use :class:`~chainer.DictSummary` to accumulate them and computes the statistics.

You can find many examples for the usage of this training utilities from the official examples. You can also search the extension implementations from :ref:`extensions`.


Trainer
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.training.Trainer

Updaters
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.training.Updater
   chainer.training.updaters.StandardUpdater
   chainer.training.updaters.ParallelUpdater
   chainer.training.updaters.MultiprocessParallelUpdater

We have two kinds of updaters for multi-gpus training. The pros/cons for the updaters are as follows:

ParallelUpdater:

* (+) Can use the same iterator for any number of GPUs
* (-) No parallelism at CPU side
* (-) GPUs used later may be blocked due to the limit of kernel-launch queue size

MultiprocessParallelUpdater:

* (+) Parallelism at CPU side
* (+) No degrade due to kernel launch queue size
* (-) Need per-process data iterator
* (-) Reporter cannot collect data except for one of the devices

.. _extensions:

Extensions
----------

An extension is a callable object that can perform arbitrary actions during the training loop.
Extensions can be registered to :class:`Trainer` by using :func:`Trainer.extend` method, and they are invoked when the :ref:`Trigger <triggers>` condition is satisfied.

In addition to the built-in extensions listed below, you can define your own extension by implementing :class:`Extension` or using the :meth:`make_extension` decorator.
See :doc:`../guides/extensions` for details.

Common
~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.training.Extension
   chainer.training.make_extension

Evaluation and Metrics Collection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These extensions provide features to collect additional metrics.
The typical use case is to use :class:`~chainer.training.extensions.Evaluator` to perform evaluation with a validation dataset to compute validation loss/accuracy.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.training.extensions.Evaluator
   chainer.training.extensions.MicroAverage

   chainer.training.extensions.FailOnNonNumber
   chainer.training.extensions.ParameterStatistics

   chainer.training.extensions.observe_lr
   chainer.training.extensions.observe_value

Optimizer Behavior Control
~~~~~~~~~~~~~~~~~~~~~~~~~~

These extensions provide features to adjust optimizer behavior.
The typical use case is to change the learning rate of the optimizer over time.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.training.extensions.ExponentialShift
   chainer.training.extensions.InverseShift
   chainer.training.extensions.LinearShift
   chainer.training.extensions.MultistepShift
   chainer.training.extensions.PolynomialShift
   chainer.training.extensions.WarmupShift
   chainer.training.extensions.StepShift

Reporting
~~~~~~~~~

These extensions provide features to perform reporting of metrics and various statistics to the console or files.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.training.extensions.PrintReport
   chainer.training.extensions.ProgressBar

   chainer.training.extensions.LogReport

   chainer.training.extensions.PlotReport
   chainer.training.extensions.VariableStatisticsPlot

   chainer.training.extensions.DumpGraph

Snapshot
~~~~~~~~

These extensions provide features to take snapshots of models.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.training.extensions.snapshot
   chainer.training.extensions.snapshot_object

Memory Release
~~~~~~~~~~~~~~

These extensions provide features to release memories.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.training.extensions.unchain_variables

.. _triggers:

Triggers
--------

A trigger is a callable object to decide when to process some specific event within the training loop. It takes a Trainer object as the argument, and returns True if some event should be fired.

It is mainly used to determine when to call an extension. It is also used to determine when to quit the training loop.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.training.get_trigger
   chainer.training.triggers.BestValueTrigger
   chainer.training.triggers.EarlyStoppingTrigger
   chainer.training.triggers.IntervalTrigger
   chainer.training.triggers.ManualScheduleTrigger
   chainer.training.triggers.MaxValueTrigger
   chainer.training.triggers.MinValueTrigger
   chainer.training.triggers.OnceTrigger
   chainer.training.triggers.TimeTrigger

