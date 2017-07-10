from chainer.training.triggers import interval_trigger
from chainer.training import util


# For backward compatibility
IntervalTrigger = interval_trigger.IntervalTrigger
get_trigger = util.get_trigger
_never_fire_trigger = util._never_fire_trigger
