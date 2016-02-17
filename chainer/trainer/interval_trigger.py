class IntervalTrigger(object):

    """Trigger based on a fixed interval.

    TODO(beam2d): document it.

    """
    def __init__(self, period, unit):
        self.period = period
        assert unit == 'epoch' or unit == 'iteration'
        self.unit = unit

    def __call__(self, trainer):
        if self.unit == 'epoch':
            return trainer.new_epoch and trainer.epoch % self.period == 0
        else:
            return trainer.t % self.period == 0
