import copy
import datetime
import warnings

import six

from chainer import backend
from chainer import configuration
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import function
from chainer import iterators
from chainer import link
from chainer import reporter as reporter_module
from chainer.training import extension
from chainer.training.extensions import util
from chainer.utils import argument


class Evaluator(extension.Extension):

    """__init__(self, iterator, target, converter=convert.concat_examples, \
device=None, eval_hook=None, eval_func=None, *, progress_bar=False)

    Trainer extension to evaluate models on a validation set.

    This extension evaluates the current models by a given evaluation function.
    It creates a :class:`~chainer.Reporter` object to store values observed in
    the evaluation function on each iteration. The report for all iterations
    are aggregated to :class:`~chainer.DictSummary`. The collected mean values
    are further reported to the reporter object of the trainer, where the name
    of each observation is prefixed by the evaluator name. See
    :class:`~chainer.Reporter` for details in naming rules of the reports.

    Evaluator has a structure to customize similar to that of
    :class:`~chainer.training.updaters.StandardUpdater`.
    The main differences are:

    - There are no optimizers in an evaluator. Instead, it holds links
      to evaluate.
    - An evaluation loop function is used instead of an update function.
    - Preparation routine can be customized, which is called before each
      evaluation. It can be used, e.g., to initialize the state of stateful
      recurrent networks.

    There are two ways to modify the evaluation behavior besides setting a
    custom evaluation function. One is by setting a custom evaluation loop via
    the ``eval_func`` argument. The other is by inheriting this class and
    overriding the :meth:`evaluate` method. In latter case, users have to
    create and handle a reporter object manually. Users also have to copy the
    iterators before using them, in order to reuse them at the next time of
    evaluation. In both cases, the functions are called in testing mode
    (i.e., ``chainer.config.train`` is set to ``False``).

    This extension is called at the end of each epoch by default.

    Args:
        iterator: Dataset iterator for the validation dataset. It can also be
            a dictionary of iterators. If this is just an iterator, the
            iterator is registered by the name ``'main'``.
        target: Link object or a dictionary of links to evaluate. If this is
            just a link object, the link is registered by the name ``'main'``.
        converter: Converter function to build input arrays.
            :func:`~chainer.dataset.concat_examples` is used by default.
        device: Device to which the validation data is sent. Negative value
            indicates the host memory (CPU).
        eval_hook: Function to prepare for each evaluation process. It is
            called at the beginning of the evaluation. The evaluator extension
            object is passed at each call.
        eval_func: Evaluation function called at each iteration. The target
            link to evaluate as a callable is used by default.
        progress_bar: Boolean flag to show a progress bar while training,
            which is similar to
            :class:`~chainer.training.extensions.ProgressBar`.
            (default: ``False``)

    .. warning::

        The argument ``progress_bar`` is experimental.
        The interface can change in the future.

    Attributes:
        converter: Converter function.
        device: Device to which the validation data is sent.
        eval_hook: Function to prepare for each evaluation process.
        eval_func: Evaluation function called at each iteration.

    """
    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = extension.PRIORITY_WRITER

    name = None

    def __init__(self, iterator, target, converter=convert.concat_examples,
                 device=None, eval_hook=None, eval_func=None, **kwargs):
        progress_bar, = argument.parse_kwargs(kwargs, ('progress_bar', False))

        if device is not None:
            device = backend.get_device(device)

        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator

        if isinstance(target, link.Link):
            target = {'main': target}
        self._targets = target

        self.converter = converter
        self.device = device
        self.eval_hook = eval_hook
        self.eval_func = eval_func

        self._progress_bar = progress_bar

        for key, iter in six.iteritems(iterator):
            if (isinstance(iter, (iterators.SerialIterator,
                                  iterators.MultiprocessIterator,
                                  iterators.MultithreadIterator)) and
                    getattr(iter, 'repeat', False)):
                msg = 'The `repeat` property of the iterator {} '
                'is set to `True`. Typically, the evaluator sweeps '
                'over iterators until they stop, '
                'but as the property being `True`, this iterator '
                'might not stop and evaluation could go into '
                'an infinite loop. '
                'We recommend to check the configuration '
                'of iterators'.format(key)
                warnings.warn(msg)

    def get_iterator(self, name):
        """Returns the iterator of the given name."""
        return self._iterators[name]

    def get_all_iterators(self):
        """Returns a dictionary of all iterators."""
        return dict(self._iterators)

    def get_target(self, name):
        """Returns the target link of the given name."""
        return self._targets[name]

    def get_all_targets(self):
        """Returns a dictionary of all target links."""
        return dict(self._targets)

    def __call__(self, trainer=None):
        """Executes the evaluator extension.

        Unlike usual extensions, this extension can be executed without passing
        a trainer object. This extension reports the performance on validation
        dataset using the :func:`~chainer.report` function. Thus, users can use
        this extension independently from any trainer by manually configuring
        a :class:`~chainer.Reporter` object.

        Args:
            trainer (~chainer.training.Trainer): Trainer object that invokes
                this extension. It can be omitted in case of calling this
                extension manually.

        Returns:
            dict: Result dictionary that contains mean statistics of values
            reported by the evaluation function.

        """
        # set up a reporter
        reporter = reporter_module.Reporter()
        if self.name is not None:
            prefix = self.name + '/'
        else:
            prefix = ''
        for name, target in six.iteritems(self._targets):
            reporter.add_observer(prefix + name, target)
            reporter.add_observers(prefix + name,
                                   target.namedlinks(skipself=True))

        with reporter:
            with configuration.using_config('train', False):
                result = self.evaluate()

        reporter_module.report(result)
        return result

    def evaluate(self):
        """Evaluates the model and returns a result dictionary.

        This method runs the evaluation loop over the validation dataset. It
        accumulates the reported values to :class:`~chainer.DictSummary` and
        returns a dictionary whose values are means computed by the summary.

        Note that this function assumes that the main iterator raises
        ``StopIteration`` or code in the evaluation loop raises an exception.
        So, if this assumption is not held, the function could be caught in
        an infinite loop.

        Users can override this method to customize the evaluation routine.

        .. note::

            This method encloses :attr:`eval_func` calls with
            :func:`function.no_backprop_mode` context, so all calculations
            using :class:`~chainer.FunctionNode`\\s inside
            :attr:`eval_func` do not make computational graphs. It is for
            reducing the memory consumption.

        Returns:
            dict: Result dictionary. This dictionary is further reported via
            :func:`~chainer.report` without specifying any observer.

        """
        iterator = self._iterators['main']
        eval_func = self.eval_func or self._targets['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            warnings.warn(
                'This iterator does not have the reset method. Evaluator '
                'copies the iterator instead of resetting. This behavior is '
                'deprecated. Please implement the reset method.',
                DeprecationWarning)
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        if self._progress_bar:
            pbar = _IteratorProgressBar(iterator=it)

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                in_arrays = convert._call_converter(
                    self.converter, batch, self.device)
                with function.no_backprop_mode():
                    if isinstance(in_arrays, tuple):
                        eval_func(*in_arrays)
                    elif isinstance(in_arrays, dict):
                        eval_func(**in_arrays)
                    else:
                        eval_func(in_arrays)

            summary.add(observation)

            if self._progress_bar:
                pbar.update()

        if self._progress_bar:
            pbar.close()

        return summary.compute_mean()

    def finalize(self):
        """Finalizes the evaluator object.

        This method calls the `finalize` method of each iterator that
        this evaluator has.
        It is called at the end of training loops.

        """
        for iterator in six.itervalues(self._iterators):
            iterator.finalize()


class _IteratorProgressBar(util.ProgressBar):

    def __init__(self, iterator, bar_length=None, out=None):
        if not (hasattr(iterator, 'current_position') and
                hasattr(iterator, 'epoch_detail')):
            raise TypeError('Iterator must have the following attributes '
                            'to enable a progress bar: '
                            'current_position, epoch_detail')
        self._iterator = iterator

        super(_IteratorProgressBar, self).__init__(
            bar_length=bar_length, out=out)

    def get_lines(self):
        iteration = self._iterator.current_position
        epoch_detail = self._iterator.epoch_detail
        epoch_size = getattr(self._iterator, '_epoch_size', None)

        lines = []

        rate = epoch_detail
        marks = '#' * int(rate * self._bar_length)
        lines.append('validation [{}{}] {:6.2%}\n'.format(
                     marks, '.' * (self._bar_length - len(marks)), rate))

        if epoch_size:
            lines.append('{:10} / {} iterations\n'
                         .format(iteration, epoch_size))
        else:
            lines.append('{:10} iterations\n'.format(iteration))

        speed_t, speed_e = self.update_speed(iteration, epoch_detail)
        estimated_time = (1.0 - epoch_detail) / speed_e
        lines.append('{:10.5g} iters/sec. Estimated time to finish: {}.\n'
                     .format(speed_t,
                             datetime.timedelta(seconds=estimated_time)))
        return lines
