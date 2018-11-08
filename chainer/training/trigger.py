import functools


class _IterationAwareTriggerWrapper(object):

    _prev_epoch_detail = None
    _prev_result = None

    def __init__(self, original_trigger):
        self._original_trigger = original_trigger

    def __call__(self, trainer):
        prev_epoch_detail = self._prev_epoch_detail
        epoch_detail = trainer.updater.epoch_detail
        if prev_epoch_detail == epoch_detail:
            result = self._prev_result
        else:
            result = self._original_trigger(trainer)
            self._prev_result = result
            self._prev_epoch_detail = epoch_detail
        return result


def iteration_aware(_arg=None):
    """iteration_aware()
Utility to make a trigger aware of training iterations.

    This utility makes the trigger cache its evaluation and return the
    same value during each iteration. This enables a single trigger instance to
    be used for multiple purposes, e.g. multiple trainer extensions.

    If the original trigger returns the same result within any given iteration,
    you don't need to use this utility.

    There are two ways to use this utility: `decorator-style` and
    `function-style`.

    Decorator-style notation is used to make a trigger class iteration-aware.

    .. code-block:: python

        # decorator-style

        @iteration_aware()
        class RandomIterationTrigger(object):
            # An iteration-aware trigger that triggers randomly at 10% of
            # all the iterations.
            def __call__(self, trainer):
                return random.randint(0, 9) == 0

    Function-style notation is used to make a single trigger instance
    iteration-aware. The original trigger can be any trigger callable.

    .. code-block:: python

        # function-style

        def random_trigger(trainer):
            # Original non-iteration-aware trigger.
            return random.randint(0, 9) == 0

        # Make the trigger iteration-aware.
        trigger = iteration_aware(random_trigger)

    """

    def wrap(cls):
        orig_call = cls.__call__

        @functools.wraps(orig_call)
        def wrapped_call(self, trainer):
            prev_epoch_detail = getattr(self, '__prev_epoch_detail', None)
            epoch_detail = trainer.updater.epoch_detail
            if prev_epoch_detail == epoch_detail:
                result = getattr(self, '__prev_result')
            else:
                result = orig_call(self, trainer)
                self.__prev_result = result
                self.__prev_epoch_detail = epoch_detail
            return result

        cls.__call__ = wrapped_call
        return cls

    if _arg is not None:
        if isinstance(_arg, type):
            # Decorator-style without parentheses, like
            #
            # @iteration_aware
            # class Trigger: ...
            cls = _arg
            return wrap(cls)

        if callable(_arg):
            # Function-style, like
            #
            # trigger = iteration_aware(original_trigger)
            orig_trigger = _arg
            return _IterationAwareTriggerWrapper(orig_trigger)

        raise TypeError('Original trigger must be a callable.')

    # Decorator-style with parentheses, like
    #
    # @iteration_aware()
    # class Trigger: ...
    return wrap
