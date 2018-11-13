class OnceTrigger(object):

    """Trigger based on the starting point of the iteration.

    This trigger accepts only once at starting point of the iteration. There
    are two ways to specify the starting point: only starting point in whole
    iteration or recalled when training resumed.

    Args:
        recall_on_resume (bool): Whether the extension is recalled or not when
            restored from a snapshot. It is set to ``False`` by default.
    """

    def __init__(self, recall_on_resume=False):
        self._recall_on_resume = recall_on_resume
        self._flag_called = False

    def trigger(self, trainer):
        if self._flag_called:
            return False
        self._flag_called = True
        return True

    @property
    def skip_initialize(self):
        """The flag decide to call `Extension.initialize` or not.

            If this flag is exist and set `True`, `Extension.initialize` is
            skipped.

        """
        return self._flag_called

    def serialize(self, serializer):
        if not self._recall_on_resume:
            self._flag_called = serializer('_flag_called', self._flag_called)
