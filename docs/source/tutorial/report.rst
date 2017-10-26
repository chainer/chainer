Customize your own logging
---------------------------

.. currentmodule:: chainer

In this section, you will learn about the following things:

* How to report logging

After reading this section, you will be able to:

* Write your own report

What is Reporter?
~~~~~~~~~~~~~~~~~~~~~~~~

:class:`chainer.Reporter` is used to collect values that users want to watch.
The reporter object manipulates a dictionary from value names to the actually
observed values. We call this dictionary as `observation`.

See the following example:

.. doctest::

    >>> from chainer import Reporter, report, report_scope
    >>>
    >>> reporter = Reporter()
    >>> observer = object()  # it can be an arbitrary (reference) object
    >>> reporter.add_observer('my_observer:', observer)
    >>> observation = {}
    >>> with reporter.scope(observation):
    ...     reporter.report({'x': 1}, observer)
    ...
    >>> observation
    {'my_observer:x': 1}

When a value is passed to the ``reporter``, an object called ``observer`` can be
optionally attached. In this case, the name of the ``observer`` is added as the
prefix of the value name. The ``observer`` name should be registered beforehand.
Using ``reporter.scope``, you can select which ``observation`` to save the
observed values.

There are also a global API to add values:

.. doctest::

    >>> observation = {}
    >>> with report_scope(observation):
    ...     report({'x': 1}, observer)
    ...
    >>> observation
    {'my_observer:x': 1}

The most important application of :class:`~chainer.Reporter` is to report
observed values from each :class:`~chainer.Link` or :class:`~chainer.Chain`
in the training and validation procedures. :class:`~chainer.training.Trainer`
and some extensions prepare their own :class:`~chainer.Reporter` object with the
hierarchy of the target link registered as observers. We can use :func:`report`
function inside any links and chains to report the observed values
(e.g., training loss, accuracy, activation statistics, etc.).


Use report in Chain
~~~~~~~~~~~~~~~~~~~~~~~~

Use report in Link
~~~~~~~~~~~~~~~~~~~~~~~~

Naming rule for the reported values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
