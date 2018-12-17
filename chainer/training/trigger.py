class Trigger(object):
    """Base class for triggers.

    Attributes:
        cached_during_iteration (bool): If ``True``, the trigger caches the
            evaluation and returns the same value within each iteration.
            Otherwise it evaluates each time it is called. This attribute
            should be set to ``True`` if a single trigger instance is used for
            multiple purpose simultaneously (e.g. for multiple trainer
            extensions). ``True`` by default.

    There are to ways to define a trigger: to inherit directly from
    :class:`chainer.Trigger` and override :meth:`chainer.Trigger.evaluate`, or
    to use :func:`chainer.trigger` decorator.

    .. code-block:: python

        # (1) Inherit chainer.Trigger

        class RandomTrigger(chainer.Trigger):
            # A trigger that triggers randomly at 10% of all the iterations.
            def evaluate(self, trainer):
                return random.randint(0, 9) == 0

        random_trigger = RandomTrigger()

        # (2) Use a decorator

        @chainer.trigger
        def random_trigger(trainer):
            # A trigger that triggers randomly at 10% of all the iterations.
            return random.randint(0, 9) == 0

    See also:
        :func:`chainer.trigger`
    """

    cached_during_iteration = True

    # None if the result is NOT cached during an iteration.
    # Otherwise a tuple containing two elements:
    # [0] epoch detail and [1] the evaluation (True/False).
    __cache_during_iteration = None

    def evaluate(self, trainer):
        """Implements the trigger logic.

        Args:
            trainer (chainer.training.Trainer): Trainer instance.

        Returns:
            bool: The evaluation of the trigger.

        This method is meant to be overridden to implement the logic of the
        trigger. The user of the trigger should not call this method. Instead,
        use the trigger as a callable::

        .. code-block:: python

            # No
            if my_trigger.evaluate(trainer):
                ...

            # Yes
            if my_trigger(trainer):
                ...
        """
        raise NotImplementedError(
            'Trigger implementation must override `evaluate` method.')

    def __call__(self, trainer):
        """Evaluates the trigger.

        Args:
            trainer (chainer.training.Trainer): Trainer instance.

        Returns:
            bool: The evaluation of the trigger.
        """
        epoch_detail = trainer.updater.epoch_detail

        # If cached_during_iteration is True, check the cache and return its
        # value if the epoch detail matches.
        if self.cached_during_iteration:
            cache = self.__cache_during_iteration
            if cache is not None:
                cached_epoch_detail, cached_result = cache
                if epoch_detail == cached_epoch_detail:
                    return cached_result

        # Evaluate the trigger
        result = self.evaluate(trainer)

        # Cache as necessary
        if self.cached_during_iteration:
            cache = (epoch_detail, result)
            self.__cache_during_iteration = cache

        return result


def trigger(_arg=None, cached_during_iteration=True):
    """trigger(cached_during_iteration=True)
Decorator to make a trigger.

    Args:
        cached_during_iteration (bool): Corresponding to
            ``cached_during_iteration`` attribute of the
            :class:`chainer.Trigger` class.

    This decorator makes a trigger instance from a user-defined function.
    The decorated function will turn into a :class:`chainer.Trigger` class
    instance.

    See :class:`chainer.Trigger` for code examples.

    .. note::
        This decorator was introduced in v6. Until then, Custom triggers can be
        defined by bare functions without this decorator. However, as there
        were no ``cached_during_iteration`` switch, using a single trigger
        instance across multiple trainer extensions had caused inconsistent
        trigger evaluation across them.

    See also:
        :class:`chainer.Trigger`
    """

    def wrap(func):

        cdi = cached_during_iteration

        class _Trigger(Trigger):
            cached_during_iteration = cdi

            def evaluate(self, trainer):
                return func(trainer)

        return _Trigger()

    if _arg is not None:
        # without parentheses (@trigger)
        assert callable(_arg)
        return wrap(_arg)

    # with parentheses (@trigger())
    return wrap
