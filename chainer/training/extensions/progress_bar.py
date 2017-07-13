from __future__ import division
import datetime
import os
import sys
import time

from chainer.training import extension
from chainer.training.extensions import util
from chainer.training import trigger
from chainer.training.non_trainer_extensions import ProgressBarPrinter


class ProgressBar(extension.Extension):

    """Trainer extension to print a progress bar and recent training status.

    This extension prints a progress bar at every call. It watches the current
    iteration and epoch to print the bar.

    Args:
        training_length (tuple): Length of whole training. It consists of an
            integer and either ``'epoch'`` or ``'iteration'``. If this value is
            omitted and the stop trigger of the trainer is
            :class:`IntervalTrigger`, this extension uses its attributes to
            determine the length of the training.
        update_interval (int): Number of iterations to skip printing the
            progress bar.
        bar_length (int): Length of the progress bar in characters.
        out: Stream to print the bar. Standard output is used by default.

    """

    def __init__(self, training_length=None, update_interval=100,
                 bar_length=50, out=sys.stdout):
        self._training_length = training_length
        self._status_template = None
        self._update_interval = update_interval
        self._bar_length = bar_length
        self._out = out
        self._recent_timing = []
        self._progress_bar = None

    def __call__(self, trainer):
        training_length = self._training_length

        # initialize some attributes at the first call
        if training_length is None:
            t = trainer.stop_trigger
            if not isinstance(t, trigger.IntervalTrigger):
                raise TypeError(
                    'cannot retrieve the training length from %s' % type(t))
            training_length = self._training_length = t.period, t.unit

        stat_template = self._status_template
        if stat_template is None:
            stat_template = self._status_template = (
                '{0.iteration:10} iter, {0.epoch} epoch / %s %ss\n' %
                training_length)

        if self._progress_bar is None:
            self._progress_bar = ProgressBarPrinter(training_length);

        out = self._out

        iteration = trainer.updater.iteration

        # print the progress bar
        if iteration % self._update_interval == 0:
            self._progress_bar(iteration, trainer.updater.epoch_detail)