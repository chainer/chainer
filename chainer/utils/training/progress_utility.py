from __future__ import print_function
from __future__ import division
import datetime
import os
import sys
import time

from chainer.dataset import iterator as iterator_module


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

    def __init__(self, interval_seconds=1):
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


class ProgressBarUtility(object):
    """Training utility to print a progress bar and recent training status.

    This utility prints a progress bar at every call. It watches the current
    iteration and epoch to print the bar.

    Args:
        training_length (tuple): Length of whole training. It consists of an
            integer and either ``'epoch'`` or ``'iteration'``. If this value is
            omitted and the stop trigger of the trainer is
            :class:`IntervalTrigger`, this extension uses its attributes to
            determine the length of the training.
        bar_length (int): Length of the progress bar in characters.
        out: Stream to print the bar. Standard output is used by default.

    """
    #todo: There is currently code duplication between this class and ProgressBar
    # since this borrows some code from that class. Consider making ProgressBar
    # be a wrapper around this class.
    def __init__(self, training_length=None, bar_length=50,
                 out=sys.stdout):
        self._training_length = training_length
        self._status_template = None
        self._bar_length = bar_length
        self._out = out
        self._recent_timing = []

    def __call__(self, iteration, epoch):
        """

        Args:
            iteration (int): Current iteration.
            epoch (float): Float-valued epoch.
        """
        training_length = self._training_length
        length, unit = training_length
        out = self._out

        # print the progress bar
        recent_timing = self._recent_timing
        now = time.time()

        recent_timing.append((iteration, epoch, now))

        if os.name == 'nt':
            util.erase_console(0, 0)
        else:
            out.write('\033[J')

        if unit == 'iteration':
            rate = iteration / length
        else:
            rate = epoch / length

        bar_length = self._bar_length
        marks = '#' * int(rate * bar_length)
        out.write('     total [{}{}] {:6.2%}\n'.format(
            marks, '.' * (bar_length - len(marks)), rate))

        epoch_rate = epoch - int(epoch)
        marks = '#' * int(epoch_rate * bar_length)
        out.write('this epoch [{}{}] {:6.2%}\n'.format(
            marks, '.' * (bar_length - len(marks)), epoch_rate))

        data = {'iteration': iteration, 'epoch': epoch}
        status = '{} iter, {} epoch / {}\n'.format(iteration, int(epoch), training_length[0])
        out.write(status)

        old_t, old_e, old_sec = recent_timing[0]
        span = now - old_sec
        if span != 0:
            speed_t = (iteration - old_t) / span
            speed_e = (epoch - old_e) / span
        else:
            speed_t = float('inf')
            speed_e = float('inf')

        if unit == 'iteration':
            estimated_time = (length - iteration) / speed_t
        else:
            estimated_time = (length - epoch) / speed_e
        out.write('{:10.5g} iters/sec. Estimated time to finish: {}.\n'
                    .format(speed_t,
                            datetime.timedelta(seconds=estimated_time)))

        # move the cursor to the head of the progress bar
        if os.name == 'nt':
            util.set_console_cursor_position(0, -4)
        else:
            out.write('\033[4A')
        out.flush()

        if len(recent_timing) > 100:
            del recent_timing[0]

    def finalize(self):
        # delete the progress bar
        out = self._out
        if os.name == 'nt':
            util.erase_console(0, 0)
        else:
            out.write('\033[J')
        out.flush()


class ProgressUtility(TimerUtility):
    """TimerUtility that tracks progress for a training loop.

    This is a TimerUtility that also tracks the progress within a training
    loop. It is intended to reduce the amount of boilerplate code that
    is required when writing a custom training loop. An optional
    progress bar display can also be enabled in the initializer.

    Specifically, this utility keeps track of the progress within the
    current epoch, tracks when a new epoch is reached, and tracks when
    the end of the training length is reached.

    Args:
        iters_per_epoch (int): The number of iterations in one epoch.

        training_length (tuple): Length of whole training. It consists of an
            integer and either ``'epoch'`` or ``'iteration'``. Once this
            object has been called this many times, the `in_progress`
            attribute will be set to False. If this value is omitted, then
            `in_progress` will always remain True.

        enable_progress_bar (bool): If True, enable a progress bar display
        which will be refreshed at a maximum rate of `interval_seconds`.

        interval_seconds (float): This sets the maximum rate, in seconds, at
        which this object will return True when called.

    Attributes:
        epoch (int): Integer-valued number of completed sweeps over the dataset.
        epoch_detail (float): Floating point number version of the epoch. For
            example, if the iterator is at the middle of the dataset at the
            third epoch, then this value is 2.5.
        interation (int): The number of iterations which is equal to the
            number of times this object has been called so far.
        is_new_epoch (bool): ``True`` if the epoch count was incremented at the last
            update.
        speed (float): The average training speed since the start of training
            in units of iterations per second.

    """

    def __init__(self, iters_per_epoch, training_length=None,
                 enable_progress_bar=False, interval_seconds=1):
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
        self._progress_bar = ProgressBarUtility(training_length);


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
        self.epoch_detail = float(self.iteration)/float(self._iters_per_epoch)

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

    Args:
        iterator (Iterator): The epoch progress information is read from
        the supplied iterator, which will be treated as read-only by this class.

        training_length (tuple): Length of whole training. It consists of an
            integer and either ``'epoch'`` or ``'iteration'``. Once this
            object has been called this many times, the `in_progress`
            attribute will be set to False. If this value is omitted, then
            `in_progress` will always remain True.

        interval_seconds (float): This sets the maximum rate, in seconds, at
        which this object will return True when called. This is also the rate
        at which the progress bar display will be updated.

    Attributes:
        interation (int): The number of iterations which is equal to the
            number of times this object has been called so far.
        speed (float): The average training speed since the start of training
            in units of iterations per second.

    """

    def __init__(self, iterator, training_length=None,
                 interval_seconds=1):
        super(IteratorProgressBar, self).__init__(interval_seconds)
        self._iterator = iterator
        self._training_length = training_length
        self._progress_bar = ProgressBarUtility(training_length);


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
