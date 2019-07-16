from __future__ import division
import os
import warnings

import numpy
import six

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer.training import extension
from chainer.training import trigger as trigger_module
from chainer.utils import argument


_available = None


def _try_import_matplotlib():
    global matplotlib, _available
    global _plot_color, _plot_color_trans, _plot_common_kwargs
    try:
        import matplotlib
        _available = True
    except ImportError:
        _available = False

    if _available:
        if hasattr(matplotlib.colors, 'to_rgba'):
            _to_rgba = matplotlib.colors.to_rgba
        else:
            # For matplotlib 1.x
            _to_rgba = matplotlib.colors.ColorConverter().to_rgba
        _plot_color = _to_rgba('#1f77b4')  # C0 color
        _plot_color_trans = _plot_color[:3] + (0.2,)  # apply alpha
        _plot_common_kwargs = {
            'alpha': 0.2, 'linewidth': 0, 'color': _plot_color_trans}


def _check_available():
    if _available is None:
        _try_import_matplotlib()

    if not _available:
        warnings.warn('matplotlib is not installed on your environment, '
                      'so nothing will be plotted at this time. '
                      'Please install matplotlib to plot figures.\n\n'
                      '  $ pip install matplotlib\n')


def _unpack_variables(x, memo=None):
    if memo is None:
        memo = ()
    if isinstance(x, chainer.Variable):
        memo += (x,)
    elif isinstance(x, chainer.Link):
        memo += tuple(x.params(include_uninit=True))
    elif isinstance(x, (list, tuple)):
        for xi in x:
            memo += _unpack_variables(xi)
    return memo


class Reservoir(object):

    """Reservoir sample with a fixed sized buffer."""

    def __init__(self, size, data_shape, dtype=numpy.float32):
        self.size = size
        self.data = numpy.zeros((size,) + data_shape, dtype=dtype)
        self.idxs = numpy.zeros((size,), dtype=numpy.int32)
        self.counter = 0

    def add(self, x, idx=None):
        if self.counter < self.size:
            self.data[self.counter] = x
            self.idxs[self.counter] = idx or self.counter
        elif self.counter >= self.size and \
                numpy.random.random() < self.size / float(self.counter + 1):
            i = numpy.random.randint(self.size)
            self.data[i] = x
            self.idxs[i] = idx or self.counter
        self.counter += 1

    def get_data(self):
        idxs = self.idxs[:min(self.counter, self.size)]
        sorted_args = numpy.argsort(idxs)
        return idxs[sorted_args], self.data[sorted_args]


class Statistician(object):

    """Helper to compute basic NumPy-like statistics."""

    def __init__(self, collect_mean, collect_std, percentile_sigmas):
        self.collect_mean = collect_mean
        self.collect_std = collect_std
        self.percentile_sigmas = percentile_sigmas

    def __call__(self, x, axis=0, dtype=None, xp=None):
        if axis is None:
            axis = tuple(range(x.ndim))
        elif not isinstance(axis, (tuple, list)):
            axis = axis,

        return self.collect(x, axis)

    def collect(self, x, axis):
        out = dict()

        if self.collect_mean:
            out['mean'] = x.mean(axis=axis)

        if self.collect_std:
            out['std'] = x.std(axis=axis)

        if self.percentile_sigmas:
            xp = backend.get_array_module(x)
            p = xp.percentile(x, self.percentile_sigmas, axis=axis)
            out['percentile'] = p

        return out


class VariableStatisticsPlot(extension.Extension):

    """__init__(\
targets, max_sample_size=1000, report_data=True, report_grad=True, \
plot_mean=True, plot_std=True, \
percentile_sigmas=(0, 0.13, 2.28, 15.87, 50, 84.13, 97.72, 99.87, 100), \
trigger=(1, 'epoch'), filename='statistics.png', figsize=None, marker=None, \
grid=True)

    Trainer extension to plot statistics for :class:`~chainer.Variable`\\s.


    This extension collects statistics for a single :class:`Variable`, a list
    of :class:`Variable`\\s or similarly a single or a list of
    :class:`Link`\\s containing one or more :class:`Variable`\\s. In case
    multiple :class:`Variable`\\s are found, the means are computed. The
    collected statistics are plotted and saved as an image in the directory
    specified by the :class:`Trainer`.

    Statistics include mean, standard deviation and percentiles.

    This extension uses reservoir sampling to preserve memory, using a fixed
    size running sample. This means that collected items in the sample are
    discarded uniformly at random when the number of items becomes larger
    than the maximum sample size, but each item is expected to occur in the
    sample with equal probability.

    Args:
        targets (:class:`Variable`, :class:`Link` or list of either):
            Parameters for which statistics are collected.
        max_sample_size (int):
            Maximum number of running samples.
        report_data (bool):
            If ``True``, data (e.g. weights) statistics are plotted.  If
            ``False``, they are neither computed nor plotted.
        report_grad (bool):
            If ``True``, gradient statistics are plotted. If ``False``, they
            are neither computed nor plotted.
        plot_mean (bool):
            If ``True``, means are plotted.  If ``False``, they are
            neither computed nor plotted.
        plot_std (bool):
            If ``True``, standard deviations are plotted.  If ``False``, they
            are neither computed nor plotted.
        percentile_sigmas (float or tuple of floats):
            Percentiles to plot in the range :math:`[0, 100]`.
        trigger:
            Trigger that decides when to save the plots as an image.  This is
            distinct from the trigger of this extension itself. If it is a
            tuple in the form ``<int>, 'epoch'`` or ``<int>, 'iteration'``, it
            is passed to :class:`IntervalTrigger`.
        filename (str):
            Name of the output image file under the output directory.
            For historical reasons ``file_name`` is also accepted as an alias
            of this argument.
        figsize (tuple of int):
            Matlotlib ``figsize`` argument that specifies the size of the
            output image.
        marker (str):
            Matplotlib ``marker`` argument that specified the marker style of
            the plots.
        grid (bool):
            Matplotlib ``grid`` argument that specifies whether grids are
            rendered in in the plots or not.
    """

    def __init__(self, targets, max_sample_size=1000,
                 report_data=True, report_grad=True,
                 plot_mean=True, plot_std=True,
                 percentile_sigmas=(
                     0, 0.13, 2.28, 15.87, 50, 84.13, 97.72, 99.87, 100),
                 trigger=(1, 'epoch'), filename=None,
                 figsize=None, marker=None, grid=True, **kwargs):

        file_name, = argument.parse_kwargs(
            kwargs, ('file_name', 'statistics.png')
        )
        if filename is None:
            filename = file_name
        del file_name  # avoid accidental use

        self._vars = _unpack_variables(targets)
        if not self._vars:
            raise ValueError(
                'Need at least one variables for which to collect statistics.'
                '\nActual: 0 <= 0')

        if not any((plot_mean, plot_std, bool(percentile_sigmas))):
            raise ValueError('Nothing to plot')

        self._keys = []
        if report_data:
            self._keys.append('data')
        if report_grad:
            self._keys.append('grad')

        self._report_data = report_data
        self._report_grad = report_grad

        self._statistician = Statistician(
            collect_mean=plot_mean, collect_std=plot_std,
            percentile_sigmas=percentile_sigmas)

        self._plot_mean = plot_mean
        self._plot_std = plot_std
        self._plot_percentile = bool(percentile_sigmas)

        self._trigger = trigger_module.get_trigger(trigger)
        self._filename = filename
        self._figsize = figsize
        self._marker = marker
        self._grid = grid

        if not self._plot_percentile:
            n_percentile = 0
        else:
            if not isinstance(percentile_sigmas, (list, tuple)):
                n_percentile = 1  # scalar, single percentile
            else:
                n_percentile = len(percentile_sigmas)
        self._data_shape = (
            len(self._keys), int(plot_mean) + int(plot_std) + n_percentile)
        self._samples = Reservoir(max_sample_size, data_shape=self._data_shape)

    @staticmethod
    def available():
        _check_available()
        return _available

    def __call__(self, trainer):
        if self.available():
            # Dynamically import pyplot to call matplotlib.use()
            # after importing chainer.training.extensions
            import matplotlib.pyplot as plt
        else:
            return

        xp = backend.get_array_module(self._vars[0].data)
        stats = xp.zeros(self._data_shape, dtype=xp.float32)
        for i, k in enumerate(self._keys):
            xs = []
            for var in self._vars:
                x = getattr(var, k, None)
                if x is not None:
                    xs.append(x.ravel())
            if xs:
                stat_dict = self._statistician(
                    xp.concatenate(xs, axis=0), axis=0, xp=xp)
                stat_list = []
                if self._plot_mean:
                    stat_list.append(xp.atleast_1d(stat_dict['mean']))
                if self._plot_std:
                    stat_list.append(xp.atleast_1d(stat_dict['std']))
                if self._plot_percentile:
                    stat_list.append(xp.atleast_1d(stat_dict['percentile']))
                stats[i] = xp.concatenate(stat_list, axis=0)

        if xp == cuda.cupy:
            stats = cuda.to_cpu(stats)
        self._samples.add(stats, idx=trainer.updater.iteration)

        if self._trigger(trainer):
            file_path = os.path.join(trainer.out, self._filename)
            self.save_plot_using_module(file_path, plt)

    def save_plot_using_module(self, file_path, plt):
        nrows = int(self._plot_mean or self._plot_std) \
            + int(self._plot_percentile)
        ncols = len(self._keys)

        fig, axes = plt.subplots(
            nrows, ncols, figsize=self._figsize, sharex=True)

        if not isinstance(axes, numpy.ndarray):  # single subplot
            axes = numpy.asarray([axes])
        if nrows == 1:
            axes = axes[None, :]
        elif ncols == 1:
            axes = axes[:, None]
        assert axes.ndim == 2

        idxs, data = self._samples.get_data()

        # Offset to access percentile data from `data`
        offset = int(self._plot_mean) + int(self._plot_std)
        n_percentile = data.shape[-1] - offset
        n_percentile_mid_floor = n_percentile // 2
        n_percentile_odd = n_percentile % 2 == 1

        for col in six.moves.range(ncols):
            row = 0
            ax = axes[row, col]
            ax.set_title(self._keys[col])  # `data` or `grad`

            if self._plot_mean or self._plot_std:
                if self._plot_mean and self._plot_std:
                    ax.errorbar(
                        idxs, data[:, col, 0], data[:, col, 1],
                        color=_plot_color, ecolor=_plot_color_trans,
                        label='mean, std', marker=self._marker)
                else:
                    if self._plot_mean:
                        label = 'mean'
                    elif self._plot_std:
                        label = 'std'
                    ax.plot(
                        idxs, data[:, col, 0], color=_plot_color, label=label,
                        marker=self._marker)
                row += 1

            if self._plot_percentile:
                ax = axes[row, col]
                for i in six.moves.range(n_percentile_mid_floor + 1):
                    if n_percentile_odd and i == n_percentile_mid_floor:
                        # Enters at most once per sub-plot, in case there is
                        # only a single percentile to plot or when this
                        # percentile is the mid percentile and the number of
                        # percentiles are odd
                        ax.plot(
                            idxs, data[:, col, offset + i], color=_plot_color,
                            label='percentile', marker=self._marker)
                    else:
                        if i == n_percentile_mid_floor:
                            # Last percentiles and the number of all
                            # percentiles are even
                            label = 'percentile'
                        else:
                            label = '_nolegend_'
                        ax.fill_between(
                            idxs,
                            data[:, col, offset + i],
                            data[:, col, -i - 1],
                            label=label,
                            **_plot_common_kwargs)
                    ax.set_xlabel('iteration')

        for ax in axes.ravel():
            ax.legend()
            if self._grid:
                ax.grid()
                ax.set_axisbelow(True)

        fig.savefig(file_path)
        plt.close()
