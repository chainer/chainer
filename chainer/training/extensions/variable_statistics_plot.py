import numpy
import os

import chainer
from chainer import cuda
from chainer.training import extension
from chainer.training import trigger as trigger_module


try:
    import matplotlib  # NOQA
    import matplotlib.pyplot as plt  # NOQA
    matplotlib_color_c0 = matplotlib.colors.to_rgba('#1f77b4')
    # matplotlib_color_c1 = matplotlib.colors.to_rgba('#ff7f0e')
    # chainer_color = matplotlib.colors.to_rgba('#e60312')
    plot_color = matplotlib_color_c0
    plot_color_trans = matplotlib_color_c0[:3] + (0.2,)
    plot_common_kwargs = {
        'alpha': 0.2, 'linewidth': 0, 'color': plot_color_trans}
    _available = True


except (ImportError, TypeError):
    _available = False


def _get_variables(x, memo=None):
    if memo is None:
        memo = ()
    if isinstance(x, chainer.Variable):
        memo += (x,)
    elif isinstance(x, chainer.Link):
        memo += tuple(x.params(include_uninit=True))
    elif isinstance(x, (list, tuple)):
        for xi in x:
            memo += _get_variables(xi)
    return memo


class Reservoir(object):

    """Reservoir sample set with a fixed sized buffer."""

    def __init__(self, size, data_shape, dtype='f'):
        self.size = size
        self.data = numpy.zeros((size, *data_shape), dtype=dtype)
        self.idxs = numpy.zeros((size,), dtype='i')
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

    percentiles = (0.13, 2.28, 15.87, 50, 84.13, 97.72, 99.87)

    def __call__(self, x, axis=0, xp=None):
        if not isinstance(axis, (tuple, list)):
            axis = axis,
        shape = (self.data_size,)
        for i, dim in enumerate(x.shape):
            if i not in axis:
                shape += dim,

        xp = xp or cuda.get_array_module(x)
        ret = xp.empty(shape, dtype=xp.float32)
        ret[0] = x.mean(axis=axis)
        ret[1] = x.std(axis=axis)
        ret[2] = x.min(axis=axis)
        ret[3] = x.max(axis=axis)
        if xp == numpy:
            ret[4:] = numpy.percentile(x, self.percentiles, axis=axis)
        else:
            # Copy to host since percentile() is not yet implemented in CuPy
            ret[4:] = cuda.to_gpu(
                numpy.percentile(cuda.to_cpu(x), self.percentiles, axis=axis))

        return xp.rollaxis(ret, 0, ret.ndim)

    @property
    def data_size(self):
        # mean, std, min, max and percentiles
        return 4 + len(self.percentiles)


class VariableStatisticsPlot(extension.Extension):

    """Trainer extension to plot statistics for :class:`Variable`s.

    This extension collects statistics for a single :class:`Variable`, a list
    of :class:`Variables`s or similarly a single or a list of
    :class:`Link`s containing one or more :class:`Variable`s. In case multiple
    :class:`Variable`s are found, the means are computed. The collected
    statistics are plotted and saved as an image in the directory specified by
    the :class:`Trainer`.

    Statistics include mean, standard deviation, minimum, maximum and
    percentiles.

    This extension uses reservoir sampling to preserve memory, using a
    fixed size sample. This means that collected items in the sample are
    discarded uniformly at random when the number of items becomes larger
    than the maximum sample size, but each item is expected to occur in the
    sample with equal probability.

    Args:
        targets (:class:`Variable`, :class:`Link` or list of either):
            Parameters for which statistics are collected.
        max_sample_size (int): Maximum number of samples.
        report_data (bool):  If ``True``, data statistics are plotted.
            If ``False``, they are neither computed or plotted.
        report_grad (bool):  If ``True``, gradient statistics are plotted.
            If ``False``, they are neither computed or plotted.
        trigger: Trigger that decides when to save the plots as an image.
            This is distinct from the trigger of this extension
            itself. If it is a tuple in the form ``<int>, 'epoch'`` or
            ``<int>, 'iteration'``, it is passed to :class:`IntervalTrigger`.
        file_name (str): Name of the plot file under the output directory.
        figsize (tuple of int): Matlotlib ``figsize`` argument that specifies
            the size of the output image.
        marker (str): Matplotlib ``marker`` argument that specified the marker
            style of the plots.
        grid (bool): Matplotlib ``grid`` argument that specifies whether grids
            are rendered in in the plots or not.
    """

    def __init__(self, targets, max_sample_size=1000,
                 report_data=True, report_grad=True,
                 trigger=(1, 'epoch'), file_name='statistics.png',
                 figsize=None, marker=None, grid=True):

        if file_name is None:
            msg = 'Missing output file name of statstics plot'
            raise ValueError(msg)

        self._vars = _get_variables(targets)
        if len(self._vars) == 0:
            msg = 'Need > 0 variables for which to collect statistics.\n' \
                  'Actual: 0'
            raise ValueError(msg)

        self._report_data = report_data
        self._report_grad = report_grad
        self._trigger = trigger_module.get_trigger(trigger)

        self._file_name = file_name
        self._figsize = figsize
        self._marker = marker
        self._grid = grid

        self._keys = []
        if report_data:
            self._keys.append('data')
        if report_grad:
            self._keys.append('grad')

        self._statistician = Statistician()
        self._data_size = self._statistician.data_size
        self._samples = Reservoir(
            max_sample_size, data_shape=(len(self._keys), self._data_size))

    def __call__(self, trainer):
        if not _available:
            # This extension does nothing if matplotlib is not available
            return

        xp = cuda.get_array_module(self._vars[0].data)

        stats = None
        for var in self._vars:
            xs = list(getattr(var, k)[None, :] for k in self._keys)
            xs = xp.concatenate(xs, axis=0)
            stat = self._statistician(
                xs, axis=tuple(range(1, xs.ndim)), xp=xp)
            if stats is None:
                stats = stat
            else:
                stats += stat
        stats = cuda.to_cpu(stats) / len(self._vars)
        self._samples.add(stats, idx=trainer.updater.iteration)

        if self._trigger(trainer):
            file_path = os.path.join(trainer.out, self._file_name)
            self.save_plot(file_path)

    def save_plot(self, file_path):
        nrows = 2  # a mean, std plot and a percentiles plot
        ncols = len(self._keys)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=self._figsize, sharex=True)
        if axes.ndim == 1:
            axes = axes[:, None]

        idxs, data = self._samples.get_data()

        for i in range(ncols):
            ax = axes[0, i]
            ax.errorbar(
                idxs, data[:, i, 0], data[:, i, 1],
                color=plot_color, ecolor=plot_color_trans,
                label='mean, std', marker=self._marker)
            ax.set_title(self._keys[i])

            ax = axes[1, i]
            ax.fill_between(
                idxs, data[:, i, 2], data[:, i, 3], **plot_common_kwargs)
            ax.fill_between(
                idxs, data[:, i, 4], data[:, i, -1], **plot_common_kwargs)
            ax.fill_between(
                idxs, data[:, i, 5], data[:, i, -2], **plot_common_kwargs)
            ax.fill_between(
                idxs, data[:, i, 6], data[:, i, -3], **plot_common_kwargs)
            ax.plot(
                idxs, data[:, i, 7], color=plot_color,
                label='percentiles', marker=self._marker)
            ax.set_xlabel('iteration')

        for ax in axes.ravel():
            ax.legend()
            if self._grid:
                ax.grid()
                ax.set_axisbelow(True)

        fig.savefig(file_path)
        plt.close()
