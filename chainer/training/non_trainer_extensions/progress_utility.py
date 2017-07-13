from __future__ import print_function
from __future__ import division
import time

from chainer.training.extensions import ProgressBarPrinter
from chainer import utils


class TimerUtility(object):

    """Timer utility

    This is a callable utility that will always return True if called at a
    rate slower than the specified interval. However, if it is called at a
    faster rate, it will only return True at a maximum rate of once every
    `interval_seconds` seconds.

    This functionality can be used, for example, when printing information
    to the console inside a training loop so as to limit the rate at which
    information is displayed to the user.

    Args:
        interval_seconds (float): The time interval corresponding to the
            maximum rate at which this object can return True when called.

    Attributes:
        iteration: The number of times this object has been called. The
            initial value is 0.
        speed: The average number of times per second this object has been
            called since it was created.
    """

    def __init__(self, interval_seconds=2):
        utils.experimental('chainer.training.non_trainer_extensions.TimerUtility')  # NOQA
        self.iteration = 0
        self.speed = 0

        self._start_time = time.time()
        self._interval_seconds = interval_seconds
        self._last_time = time.time()

    def __call__(self):
        """Return True or False depending on elapsed time since last call.

        This method should be called once each iteration so that the
        `iteration` and `speed` attributes will be updated appropriately.

        If at least `interval_seconds` has elapsed since the last call, return
        True. Otherwise return False.
        """
        self.iteration += 1
        ret_val = False
        last_time = self._last_time
        now = time.time()
        delta_time = now - last_time
        self.speed = float(self.iteration) / (now - self._start_time)
        if delta_time >= self._interval_seconds:
            self._last_time = now
            ret_val = True

        return ret_val


class ProgressUtility(TimerUtility):

    """TimerUtility that tracks progress for a training loop.

    This is a TimerUtility that also tracks the progress within a training
    loop. It is intended to reduce the amount of boilerplate code that
    is required when writing a custom training loop. An optional
    progress bar display can also be enabled in the initializer.

    Specifically, this utility keeps track of the progress within the
    current epoch, when a new epoch is reached, and when
    the end of the training is reached.

    Usage:
        This utility is intended to be used to display a progress bar
        and track epoch/iteration progress in a cutom training loop
        (that is not using Trainer) in which a dataset iterator is
        not used. If a dataset iterator is used then
        `IteratorProgressBar` may be more appropriate.

    Args:
        iters_per_epoch (int): The number of iterations in one epoch.

        training_length (tuple): Length of whole training. It consists of an
            integer and either ``'epoch'`` or ``'iteration'``.

        enable_progress_bar (bool): If True, enable a progress bar display
        that will be refreshed at a maximum rate of `interval_seconds`.

        interval_seconds (float): This sets the maximum rate, in seconds, at
        which this object will return True when called.

    Attributes:
        epoch (int): Integer-valued number of completed sweeps over the
            dataset.
        epoch_detail (float): Floating point number version of the epoch. For
            example, if the iterator is at the middle of the dataset at the
            third epoch, then this value is 2.5.
        interation (int): The number of iterations which is equal to the
            number of times this object has been called so far.
        is_new_epoch (bool): ``True`` if the epoch count was incremented at
            the last update.

    """

    def __init__(self, iters_per_epoch, training_length,
                 enable_progress_bar=False, interval_seconds=2):
        super(ProgressUtility, self).__init__(interval_seconds)
        # public attributes:
        # int-valued epoch
        self.epoch = 0
        # float-valued epoch
        self.epoch_detail = 0
        # True if the most recent call to update() was the
        # last call in the current epoch.
        self.is_new_epoch = False

        # private attribute:
        self._iters_per_epoch = iters_per_epoch
        self._training_length = training_length
        self._enable_progress_bar = enable_progress_bar
        self._progress_bar = ProgressBarPrinter(training_length)

    def __call__(self):
        """Return True or False depending on elapsed time since last call.

        This method should be called once each iteration.

        If at least `interval_seconds` has elapsed since the last call, return
        True. Otherwise return False.
        """
        if self.iteration % self._iters_per_epoch == 0 and self.iteration != 0:
            self.is_new_epoch = True
            self.epoch += 1
        else:
            self.is_new_epoch = False
        self.epoch_detail = float(self.iteration) /\
            float(self._iters_per_epoch)

        ret_val = super(ProgressUtility, self).__call__()
        if ret_val:
            if self._enable_progress_bar:
                self._progress_bar(self.iteration, self.epoch_detail)

        return ret_val


class IteratorProgressBar(TimerUtility):

    """TimerUtility that displays a progress bar for a training loop.

    This is a TimerUtility that displays a progress bar using information
    from a supplied iterator. The progress bar display is updated at the
    same rate that `__call__()` returns True.

    Usage:
        This utility is intended to be used to display a progress bar
        in a custom training loop (that is not using Trainer) in which
        a dataset iterator is also used.

    Args:
        iterator (Iterator): The epoch progress information is read from
        the supplied iterator.

        training_length (tuple): Length of whole training. It consists of an
            integer and either ``'epoch'`` or ``'iteration'``.

        interval_seconds (float): This sets the maximum rate, in seconds, at
        which this object will return True when called. This is also the rate
        at which the progress bar display will be updated.

    """

    def __init__(self, iterator, training_length,
                 interval_seconds=2):
        super(IteratorProgressBar, self).__init__(interval_seconds)
        self._iterator = iterator
        self._training_length = training_length
        self._progress_bar = ProgressBarPrinter(training_length)

    def __call__(self):
        """Return True or False depending on elapsed time since last call.

        This method should be called once each iteration.
        For the iterator that was supplied to the initializer, it is also
        assumed that iterator.next() is called once per iteration and is
        always called before calling this method.

        If at least `interval_seconds` has elapsed since the last call, return
        True and update the progress bar display. Otherwise return False.
        """
        ret_val = super(IteratorProgressBar, self).__call__()
        if ret_val:
            self._progress_bar(self.iteration, self._iterator.epoch_detail)

        return ret_val
