class IntervalTrigger(Trigger):

    """Trigger based on a fixed interval.

    TODO(beam2d): document it.

    """
    def __init__(self, period, unit):
        self.period = period
        assert unit == 'epoch' or unit == 'iteration'
        self.unit = unit

    def __call__(self, epoch, new_epoch, t, **kwargs):
        if self.unit == 'epoch':
            return new_epoch and epoch % self.period == 0
        else:
            return t % self.period == 0
