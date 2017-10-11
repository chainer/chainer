from chainer.training import extension  # NOQA
from chainer.training import extensions  # NOQA
from chainer.training import trainer  # NOQA
from chainer.training import trigger  # NOQA
from chainer.training import triggers  # NOQA
from chainer.training import updater  # NOQA
from chainer.training import updaters  # NOQA
from chainer.training import util  # NOQA


# import class and function
from chainer.training.extension import Extension  # NOQA
from chainer.training.extension import make_extension  # NOQA
from chainer.training.extension import PRIORITY_EDITOR  # NOQA
from chainer.training.extension import PRIORITY_READER  # NOQA
from chainer.training.extension import PRIORITY_WRITER  # NOQA
from chainer.training.trainer import Trainer  # NOQA
from chainer.training.trigger import get_trigger  # NOQA
from chainer.training.trigger import IntervalTrigger  # NOQA
from chainer.training.updater import ParallelUpdater  # NOQA
from chainer.training.updater import StandardUpdater  # NOQA
from chainer.training.updater import Updater  # NOQA


from chainer.configuration import config  # NOQA
from chainer.configuration import global_config  # NOQA

import numpy


global_config.update_parameter_in_fp32 = False
global_config.loss_scaling_factor = None


def set_loss_scaling_factor(val):
    """Sets loss scaling factor."""
    config.loss_scaling_factor = val
    config.update_parameter_in_fp32 = True


def get_loss_scaling_factor():
    """Gets loss scaling factor."""
    return config.loss_scaling_factor


def should_update_parameter_in_fp32(dtype):
    """Determins if the parameter should be updated in fp32.

    Args:
        dtype (numpy.dtype): data type of the parameter.

    Returns:
        bool: ``True`` if the parameter should be updated in fp32.
    """
    if config.update_parameter_in_fp32 and dtype == numpy.float16:
        return True
    return False
