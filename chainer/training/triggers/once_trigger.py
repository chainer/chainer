import warnings


class OnceTrigger(object):

    """Trigger based on the starting point of the iteration.

    This trigger accepts only once at starting point of the iteration. There
    are two ways to specify the starting point: only starting point in whole
    iteration or called again when training resumed.

    Args:
        call_on_resume (bool): Whether the extension is called again or not
            when restored from a snapshot. It is set to ``False`` by default.

    Attributes:
        finished (bool): Flag that indicates whether or not this trigger will
        fire in the future. This flag is used to determine if the extension
        should be initialized after resume.

    """

    def __init__(self, call_on_resume=False):
        self._flag_first = True
        self._flag_resumed = call_on_resume

    @property
    def finished(self):
        return not (self._flag_first or self._flag_resumed)

    def __call__(self, trainer):
        fire = not self.finished
        self._flag_resumed = False
        self._flag_first = False
        return fire

    def serialize(self, serializer):
        try:
            self._flag_first = serializer('_flag_first', self._flag_first)
        except KeyError:
            warnings.warn(
                'The flag is not saved.'
                'OnceTrigger guess it is not first when resumed. '
                'If this trigger is resumed before first called, '
                'it may not work correctly.')
            self._flag_first = False
