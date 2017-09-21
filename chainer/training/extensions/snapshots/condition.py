class Condition(object):
    """Base class of snapshot condition.

    This class represent the condition whether a snapshot should be taken or
    not. ``__call__()`` is invoked every time when
    :class:`~chainer.training.extensions.Snapshot` object's extension trigger
    is pulled. This class is usefull if you use multiple processes snapshot.
    """

    def __init__(self):
        pass

    def __call__(self, trainer, snapshot):
        """Determine the condition is met or not.

        Args:
            trainer (:class:`~chainer.training.Trainer`): Trainer object that
                invokes this operator indirectly.
            snapshot (:class:`~chainer.training.extensions.Snapshot`):
                Snapshot object that invokes this operator directly.

        Returns:
            bool: True if condition met else false.
        """
        return False


class Always(Condition):
    """Snapshot condition that always return true.

    This class always returns true for its condition. This is the default
    condition for ``Snapshot`` object.
    """

    def __call__(self, trainer, snapshot):
        return True
