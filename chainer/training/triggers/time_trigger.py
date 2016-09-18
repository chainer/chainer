import time


class TimeTrigger(object):

    """Trigger based on a fixed time interval.

    This trigger accepts iterations with a given interval time.

    Args:
        period (float): Interval time. It is given in seconds.

    """

    def __init__(self, period):
        self._period = period
        self._next_time = time.time()

    def __call__(self, trainer):
        now = time.time()
        if self._next_time < now:
            self._next_time += self._period
            return True
        else:
            return False
