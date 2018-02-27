from __future__ import division

from chainer import training


try:
    import mock
    _error = None
except ImportError as e:
    _error = e


def is_available():
    return _error is None


def check_available():
    if _error is not None:
        raise RuntimeError('''\
{} is not available.

Reason: {}: {}'''.format(__name__, type(_error).__name__, _error))


def get_error():
    return _error


def get_trainer_with_mock_updater(
        stop_trigger=(10, 'iteration'), iter_per_epoch=10, extensions=None):
    """Returns a :class:`~chainer.training.Trainer` object with mock updater.

    The returned trainer can be used for testing the trainer itself and the
    extensions. A mock object is used as its updater. The update function set
    to the mock correctly increments the iteration counts (
    ``updater.iteration``), and thus you can write a test relying on it.

    Args:
        stop_trigger: Stop trigger of the trainer.
        iter_per_epoch: The number of iterations per epoch.
        extensions: Extensions registered to the trainer.

    Returns:
        Trainer object with a mock updater.

    """
    if extensions is None:
        extensions = []
    check_available()
    updater = mock.Mock()
    updater.get_all_optimizers.return_value = {}
    updater.iteration = 0
    updater.epoch = 0
    updater.epoch_detail = 0
    updater.is_new_epoch = True
    updater.previous_epoch_detail = None

    def update():
        updater.update_core()
        updater.iteration += 1
        updater.epoch = updater.iteration // iter_per_epoch
        updater.epoch_detail = updater.iteration / iter_per_epoch
        updater.is_new_epoch = (updater.iteration - 1) // \
            iter_per_epoch != updater.epoch
        updater.previous_epoch_detail = (updater.iteration - 1) \
            / iter_per_epoch

    updater.update = update
    trainer = training.Trainer(updater, stop_trigger, extensions=extensions)
    return trainer
