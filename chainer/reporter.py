from __future__ import absolute_import
import collections
import contextlib
import copy
import json
import threading
import typing as tp  # NOQA
import warnings

import numpy
import six

import chainer
from chainer import backend
from chainer import configuration
from chainer import serializer as serializer_module
from chainer import variable
import chainerx


_thread_local = threading.local()


def _copy_variable(value):
    if isinstance(value, variable.Variable):
        return copy.copy(value)
    return value


class Reporter(object):

    """Object to which observed values are reported.

    Reporter is used to collect values that users want to watch. The reporter
    object holds a mapping from value names to the actually observed values.
    We call this mapping `observations`.

    When a value is passed to the reporter, an object called `observer` can be
    optionally attached. In this case, the name of the observer is added as the
    prefix of the value name. The observer name should be registered
    beforehand.

    See the following example::

       >>> from chainer import Reporter, report, report_scope
       >>>
       >>> reporter = Reporter()
       >>> observer = object()  # it can be an arbitrary (reference) object
       >>> reporter.add_observer('my_observer', observer)
       >>> observation = {}
       >>> with reporter.scope(observation):
       ...     reporter.report({'x': 1}, observer)
       ...
       >>> observation
       {'my_observer/x': 1}

    There are also a global API to add values::

       >>> observation = {}
       >>> with report_scope(observation):
       ...     report({'x': 1}, observer)
       ...
       >>> observation
       {'my_observer/x': 1}

    The most important application of Reporter is to report observed values
    from each link or chain in the training and validation procedures.
    :class:`~chainer.training.Trainer` and some extensions prepare their own
    Reporter object with the hierarchy of the target link registered as
    observers. We can use :func:`report` function inside any links and chains
    to report the observed values (e.g., training loss, accuracy, activation
    statistics, etc.).

    Attributes:
        observation: Dictionary of observed values.

    """

    def __init__(self):
        self._observer_names = {}
        self.observation = {}

    def __enter__(self):
        """Makes this reporter object current."""
        _get_reporters().append(self)

    def __exit__(self, exc_type, exc_value, traceback):
        """Recovers the previous reporter object to the current."""
        _get_reporters().pop()

    @contextlib.contextmanager
    def scope(self, observation):
        """Creates a scope to report observed values to ``observation``.

        This is a context manager to be passed to ``with`` statements. In this
        scope, the observation dictionary is changed to the given one.

        It also makes this reporter object current.

        Args:
            observation (dict): Observation dictionary. All observations
                reported inside of the ``with`` statement are written to this
                dictionary.

        """
        old = self.observation
        self.observation = observation
        self.__enter__()
        yield
        self.__exit__(None, None, None)
        self.observation = old

    def add_observer(self, name, observer):
        """Registers an observer of values.

        Observer defines a scope of names for observed values. Values observed
        with the observer are registered with names prefixed by the observer
        name.

        Args:
            name (str): Name of the observer.
            observer: The observer object. Note that the reporter distinguishes
                the observers by their object ids (i.e., ``id(owner)``), rather
                than the object equality.

        """
        self._observer_names[id(observer)] = name

    def add_observers(self, prefix, observers):
        """Registers multiple observers at once.

        This is a convenient method to register multiple objects at once.

        Args:
            prefix (str): Prefix of each name of observers.
            observers: Iterator of name and observer pairs.

        """
        for name, observer in observers:
            self._observer_names[id(observer)] = prefix + name

    def report(self, values, observer=None):
        """Reports observed values.

        The values are written with the key, prefixed by the name of the
        observer object if given.

        .. note::
           If a value is of type :class:`~chainer.Variable`, the
           variable is copied without preserving the computational graph and
           the new variable object purged from the graph is stored to the
           observer. This behavior can be changed by setting
           ``chainer.config.keep_graph_on_report`` to ``True``.

        Args:
            values (dict): Dictionary of observed values.
            observer: Observer object. Its object ID is used to retrieve the
                observer name, which is used as the prefix of the registration
                name of the observed value.

        """
        if not configuration.config.keep_graph_on_report:
            values = {k: _copy_variable(v) for k, v in six.iteritems(values)}

        if observer is not None:
            observer_id = id(observer)
            if observer_id not in self._observer_names:
                raise KeyError(
                    'Given observer is not registered to the reporter.')
            observer_name = self._observer_names[observer_id]
            for key, value in six.iteritems(values):
                name = '%s/%s' % (observer_name, key)
                self.observation[name] = value
        else:
            self.observation.update(values)


def _get_reporters():
    try:
        reporters = _thread_local.reporters
    except AttributeError:
        reporters = _thread_local.reporters = []
    return reporters


def get_current_reporter():
    """Returns the current reporter object."""
    return _get_reporters()[-1]


def report(values, observer=None):
    """Reports observed values with the current reporter object.

    Any reporter object can be set current by the ``with`` statement. This
    function calls the :meth:`Reporter.report` method of the current reporter.
    If no reporter object is current, this function does nothing.

    .. admonition:: Example

       The most typical example is a use within links and chains. Suppose that
       a link is registered to the current reporter as an observer (for
       example, the target link of the optimizer is automatically registered to
       the reporter of the :class:`~chainer.training.Trainer`). We can report
       some values from the link as follows::

          class MyRegressor(chainer.Chain):
              def __init__(self, predictor):
                  super(MyRegressor, self).__init__(predictor=predictor)

              def __call__(self, x, y):
                  # This chain just computes the mean absolute and squared
                  # errors between the prediction and y.
                  pred = self.predictor(x)
                  abs_error = F.sum(abs(pred - y)) / len(x)
                  loss = F.mean_squared_error(pred, y)

                  # Report the mean absolute and squared errors.
                  chainer.report({
                      'abs_error': abs_error,
                      'squared_error': loss,
                  }, self)

                  return loss

       If the link is named ``'main'`` in the hierarchy (which is the default
       name of the target link in the
       :class:`~chainer.training.updaters.StandardUpdater`),
       these reported values are
       named ``'main/abs_error'`` and ``'main/squared_error'``. If these values
       are reported inside the :class:`~chainer.training.extensions.Evaluator`
       extension, ``'validation/'`` is added at the head of the link name, thus
       the item names are changed to ``'validation/main/abs_error'`` and
       ``'validation/main/squared_error'`` (``'validation'`` is the default
       name of the Evaluator extension).

    Args:
        values (dict): Dictionary of observed values.
        observer: Observer object. Its object ID is used to retrieve the
            observer name, which is used as the prefix of the registration name
            of the observed value.

    """
    reporters = _get_reporters()
    if reporters:
        current = reporters[-1]
        current.report(values, observer)


@contextlib.contextmanager
def report_scope(observation):
    """Returns a report scope with the current reporter.

    This is equivalent to ``get_current_reporter().scope(observation)``,
    except that it does not make the reporter current redundantly.

    """
    current = _get_reporters()[-1]
    old = current.observation
    current.observation = observation
    yield
    current.observation = old


class Summary(object):

    """Online summarization of a sequence of scalars.

    Summary computes the statistics of given scalars online.

    """

    def __init__(self):
        self._x = 0.0
        self._x2 = 0.0
        self._n = 0

    def add(self, value, weight=1):
        """Adds a scalar value.

        Args:
            value: Scalar value to accumulate. It is either a NumPy scalar or
                a zero-dimensional array (on CPU or GPU).
            weight: An optional weight for the value. It is a NumPy scalar or
                a zero-dimensional array (on CPU or GPU).
                Default is 1 (integer).

        """
        if isinstance(value, chainerx.ndarray):
            # ChainerX arrays does not support inplace assignment if it's
            # connected to the backprop graph.
            value = value.as_grad_stopped()

        with chainer.using_device(backend.get_device_from_array(value)):
            self._x += weight * value
            self._x2 += weight * value * value
            self._n += weight

    def compute_mean(self):
        """Computes the mean."""
        x, n = self._x, self._n
        with chainer.using_device(backend.get_device_from_array(x)):
            return x / n

    def make_statistics(self):
        """Computes and returns the mean and standard deviation values.

        Returns:
            tuple: Mean and standard deviation values.

        """
        x, n = self._x, self._n
        xp = backend.get_array_module(x)
        with chainer.using_device(backend.get_device_from_array(x)):
            mean = x / n
            var = self._x2 / n - mean * mean
            std = xp.sqrt(var)
            return mean, std

    def serialize(self, serializer):
        try:
            self._x = serializer('_x', self._x)
            self._x2 = serializer('_x2', self._x2)
            self._n = serializer('_n', self._n)
        except KeyError:
            warnings.warn('The previous statistics are not saved.')


class DictSummary(object):

    """Online summarization of a sequence of dictionaries.

    ``DictSummary`` computes the statistics of a given set of scalars online.
    It only computes the statistics for scalar values and variables of scalar
    values in the dictionaries.

    """

    def __init__(self):
        self._summaries = collections.defaultdict(Summary)

    def add(self, d):
        """Adds a dictionary of scalars.

        Args:
            d (dict): Dictionary of scalars to accumulate. Only elements of
               scalars, zero-dimensional arrays, and variables of
               zero-dimensional arrays are accumulated. When the value
               is a tuple, the second element is interpreted as a weight.

        """
        summaries = self._summaries
        for k, v in six.iteritems(d):
            w = 1
            if isinstance(v, tuple):
                w = v[1]
                v = v[0]
                if isinstance(w, variable.Variable):
                    w = w.array
                if not numpy.isscalar(w) and not getattr(w, 'ndim', -1) == 0:
                    raise ValueError(
                        'Given weight to {} was not scalar.'.format(k))
            if isinstance(v, variable.Variable):
                v = v.array
            if numpy.isscalar(v) or getattr(v, 'ndim', -1) == 0:
                summaries[k].add(v, weight=w)

    def compute_mean(self):
        """Creates a dictionary of mean values.

        It returns a single dictionary that holds a mean value for each entry
        added to the summary.

        Returns:
            dict: Dictionary of mean values.

        """
        return {name: summary.compute_mean()
                for name, summary in six.iteritems(self._summaries)}

    def make_statistics(self):
        """Creates a dictionary of statistics.

        It returns a single dictionary that holds mean and standard deviation
        values for every entry added to the summary. For an entry of name
        ``'key'``, these values are added to the dictionary by names ``'key'``
        and ``'key.std'``, respectively.

        Returns:
            dict: Dictionary of statistics of all entries.

        """
        stats = {}
        for name, summary in six.iteritems(self._summaries):
            mean, std = summary.make_statistics()
            stats[name] = mean
            stats[name + '.std'] = std

        return stats

    def serialize(self, serializer):
        if isinstance(serializer, serializer_module.Serializer):
            names = list(self._summaries.keys())
            serializer('_names', json.dumps(names))
            for index, name in enumerate(names):
                self._summaries[name].serialize(
                    serializer['_summaries'][str(index)])
        else:
            self._summaries.clear()
            try:
                names = json.loads(serializer('_names', ''))
            except KeyError:
                warnings.warn('The names of statistics are not saved.')
                return
            for index, name in enumerate(names):
                self._summaries[name].serialize(
                    serializer['_summaries'][str(index)])
