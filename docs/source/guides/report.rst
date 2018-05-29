Customize your own logging
---------------------------

.. currentmodule:: chainer

In this section, you will learn about the following things:

* What is :class:`chainer.Reporter`?
* How to report logging with :class:`chainer.Reporter`?
* The naming rule for the reported values.

After reading this section, you will be able to:

* Write your own report.

What is Reporter?
~~~~~~~~~~~~~~~~~~

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
    {'my_observer:/x': 1}

When a value is passed to the ``reporter``, an object called ``observer`` can be
optionally attached. In this case, the name of the ``observer`` is added as the
prefix of the value name. The ``observer`` name should be registered beforehand.
Using ``reporter.scope``, you can select which ``observation`` to save the
observed values.

There are also a global API :func:`chainer.report`, which reports observed values
with the current reporter object. In this case, `current` means which ``with``
statement scope the current code line is in. This function calls the
:func:`Reporter.report()` method of the current reporter. 

.. doctest::

    >>> observation = {}
    >>> with reporter.scope(observation):
    ...     report({'x': 1}, observer)
    ...
    >>> observation
    {'my_observer:/x': 1}

Use report in Chain or Link
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The most important application of :class:`~chainer.Reporter` is to report
observed values from each :class:`~chainer.Link` or :class:`~chainer.Chain`
in the training and validation procedures.

But, how to report the observed values from each link or chain? Shold we
prepare the :class:`~chainer.Reporter`? No, you only need to call
:func:`report` in chain or link,
because :class:`~chainer.training.Trainer` and some extensions prepare their own
:class:`~chainer.Reporter` object with the hierarchy of the target link registered
as observers. We can use :func:`report` function inside any links and chains to
report the observed values (e.g., training loss, accuracy, activation statistics, etc.).

See the following example:

.. doctest::

    >>> class Classifier(Chain):
    ...     def __init__(self, predictor):
    ...         super(Classifier, self).__init__()
    ...         with self.init_scope():
    ...             self.predictor = predictor
    ...
    ...     def forward(self, x, t):
    ...         y = self.predictor(x)
    ...         loss = F.softmax_cross_entropy(y, t)
    ...         accuracy = F.accuracy(y, t)
    ...         report({'loss': loss, 'accuracy': accuracy}, self)
    ...         return loss
    ...

If the link is named ``'main'`` in the hierarchy (which is the default
name of the target link in the :class:`~chainer.training.StandardUpdater`),
these reported values are named ``'main/loss'`` and ``'main/accuracy'``.
If these values are reported inside the :class:`~chainer.training.extensions.Evaluator`
extension, ``'validation/'`` is added at the head of the link name, thus
the item names are changed to ``'validation/main/loss'`` and ``'validation/main/accuracy'``
(``'validation'`` is the default name of the Evaluator extension).

Naming rule for the reported values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

So, you know almost everything about :class:`~chainer.Reporter`.
However, there is one more thing. It is what is the naming rule for the reported values,
especially when the values are reported from a link that is not the root of the link hierarchy.

As we explained in the previous section, the root of links is named as ``'main'`` 
by the the :class:`~chainer.training.StandardUpdater` and the names of reported
values in the root have the prefix ``'main/'``.
When the values are reported from a link that is not the root of the link hierarchy,
the prefix of the names are determined by the link hierarchy, or
:func:`~chainer.Link.namedlinks`.

See the following example:

.. doctest::

    >>> class MLP(Chain):
    ...     def __init__(self, n_units, n_out):
    ...         super(MLP, self).__init__()
    ...         with self.init_scope():
    ...             # the size of the inputs to each layer will be inferred
    ...             self.l1 = L.Linear(None, n_units)  # n_in -> n_units
    ...             self.l2 = L.Linear(None, n_units)  # n_units -> n_units
    ...             self.l3 = L.Linear(None, n_out)    # n_units -> n_out
    ...
    ...     def forward(self, x):
    ...         h1 = F.relu(self.l1(x))
    ...         h2 = F.relu(self.l2(h1))
    ...         y = self.l3(h2)
    ...         report({'sum_y': F.sum(y)}, self)
    ...         return y
    ...
    >>> model = Classifier(MLP(100, 10))
    >>> for name, observer in model.namedlinks(skipself=True):
    ...     print(name)  # doctest: +SKIP
    /predictor
    /predictor/l1
    /predictor/l2
    /predictor/l3

You can get the parameters of the link hierarchy by :func:`~chainer.Link.namedlinks`.
In this example, we report ``'loss'`` and ``'accuracy'`` in the root of links, and
``'sum_y'`` in the link of ``'/predictor'``.
So, you can access the reported values by ``'main/accuracy'``,
``'main/accuracy'``, and ``'main/predictor/sum_y'``.

See what we explained is correct:

.. code-block:: console

    >>> train, test = datasets.get_mnist()
    >>> train_iter = iterators.SerialIterator(train, batch_size=100, shuffle=True)
    >>> test_iter = iterators.SerialIterator(test, batch_size=100, repeat=False, shuffle=False)
    >>> optimizer = optimizers.SGD()
    >>> optimizer.setup(model)
    >>> updater = training.StandardUpdater(train_iter, optimizer)
    >>> trainer = training.Trainer(updater, (1, 'epoch'), out='result')
    >>> trainer.extend(extensions.Evaluator(test_iter, model))
    >>> trainer.extend(extensions.LogReport())
    >>> trainer.extend(extensions.PrintReport(
    ...     ['epoch', 'main/accuracy', 'main/loss', 'main/predictor/sum_y', 'validation/main/accuracy']))
    >>> trainer.run()
    epoch       main/accuracy  main/loss   main/predictor/sum_y  validation/main/accuracy
    1           0.662317       1.38345     47.9927               0.8498    
