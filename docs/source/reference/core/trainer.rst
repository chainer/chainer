.. module:: chainer.trainer

Trainer, Updater, and Extension (experimental)
==============================================

Trainer is a framework to provide a reusable implementation of training loops. It consists of following classes:

- :class:`Trainer` class that actually implements the training loop.
- :class:`Updater` interface that determines the way to update the optimizer.
- :class:`Extension` interface that adds a functionality to the training loop.

List of builtin extensions are provided at :ref:`extensions`.

.. note::
   This feature is experimental.
   The interface may be changed in the future releases.


Trainer
-------

.. currentmodule:: chainer
.. autoclass:: Trainer
   :members:
.. autofunction:: create_standard_trainer


Updater
-------

.. currentmodule:: chainer.trainer
.. autoclass:: Updater
   :members:
.. autoclass:: StandardUpdater


Extension
---------

.. autoclass:: Extension
   :members:
.. autofunction:: make_extension

.. autoclass:: IntervalTrigger
   :members:
