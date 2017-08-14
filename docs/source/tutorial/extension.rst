Create your own trainer extension
=================================

.. currentmodule:: chainer

In this section, you will learn about the following things:

* How to create your own trainer extension
    * by defining a new simple function
    * by defining a new function decorated with :meth:`~chainer.training.make_extension`
    * by defining a new class inherited from :class:`~chainer.training.Extension`

What is trainer Extension?
--------------------------

:class:`~chainer.training.Extension` is a callable object that takes a :class:`~chainer.training.Trainer` object as an argument. Adding an :class:`~chainer.training.Extension` to a :class:`~chainer.training.Trainer` using :meth:`~chainer.training.Trainer.extend` method, the :class:`~chainer.training.Extension` will be called at given timing you specified by using ``trigger`` object (See the details below.)

A :class:`~chainer.training.Trainer` object has all information used in a training loop, e.g., models, optimizers, updaters, iterators, and datasets, etc. So you can change the settings of optimizers 


Write a simple function for a new Extension
-------------------------------------------

You can make a new :class:`~chainer.training.Extension` by writing a simple function which takes :class:`~chainer.training.Trainer` object as its argument. For example, when you want to reduce the learning rate at specified timing during training, ``lr_drop`` extension can be written as follows:

.. testcode::

    def lr_drop(trainer):
        trainer.updater.get_optimizer('main').lr *= 0.1

Then you can add this function to a :class:`~chainer.training.Trainer` object via :meth:`~chainer.training.Trainer.extend` method.

.. code-block:: python

    trainer.extend(lr_drop, trigger=(10, 'epoch'))

It performs learning rate dropping at every 10 epochs by multiplying 0.1 with the current learning rate.

:class:`~chainer.training.Trainer` also accepts a lambda function as an :class:`~chainer.training.Extension`, so the :class:`~chainer.training.Extension` above can be written as follows:

.. testcode::

    lr_drop = lambda trainer: trainer.updater.get_optimizer('main').lr *= 0.1


Write a method decorated with @make_extension
---------------------------------------------

:meth:`~chainer.training.make_extension` is a decorator that adds some attributes to a given function. For example, the simple extension we created above can be written in this form:

.. testcode::

    @make_extension(trigger=(10, 'iteration'), )


Attributes added by this decorator are follows.

1. trigger
^^^^^^^^^^

``trigger`` is an object that takes a :class:`~chainer.training.Trainer` object as an argument and returns a boolean value. If a tuple in a form ``(period, unit)`` is given as a trigger, it will be considered as an :class:`~chainer.training.triggers.IntervalTrigger` that invokes the extension at every ``period`` ``unit``. For example, when the given tuple is ``(10, 'epoch')``, the extension will be fired at every 10 epochs.

``trigger`` can also be given to the :meth:`~chainer.training.trainer.extend` method that adds an extension to a :class:`~chainer.training.Trainer` object. The priority of ``trigger``\ s is as follows:

- When both :meth:`~chainer.training.Trainer.extend` and a given :class:`~chainer.training.Extension` have ``trigger``\ s, the ``trigger`` given to :meth:`~chainer.training.Trainer.extend` is used.
- When ``None`` is given to :meth:`~chainer.training.Trainer.extend` as the ``trigger`` argument and a given :class:`~chainer.training.Extension` has ``trigger``, the ``trigger`` given to the :class:`~chainer.training.Extension` is used.
- When both ``trigger`` attributes in :meth:`~chainer.training.Trainer.extend` and :class:`~chainer.training.Extension` are ``None``, the :class:`~chainer.training.Extension` will be fired at every iteration.

See the details in the documentation of :meth:`~chainer.training.get_trigger`.

2.
