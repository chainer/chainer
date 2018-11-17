class OnceTrigger(object):

    """Trigger based on the starting point of the iteration.

    This trigger accepts only once at starting point of the iteration. There
    are two ways to specify the starting point: only starting point in whole
    iteration or called again when training resumed.

    Args:
        call_on_resume (bool): Whether the extension is called again or not
            when restored from a snapshot. It is set to ``False`` by default.

    Attributes:
        finished (bool): Flag that triggered when this trigger called once.
            The flag helps decision to call `Extension.initialize` or not
            in `trainer`.
    """

    def __init__(self, call_on_resume=False):
        self.finished = False
        self._flag_force = call_on_resume

    def trigger(self, trainer):
        flag = self._flag_force or not self.finished
        self._flag_force = False
        self.finished = True
        return flag

    def serialize(self, serializer):
        self.finished = serializer('finished', self.finished)
