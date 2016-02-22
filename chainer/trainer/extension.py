PRIORITY_WRITER = 300
PRIORITY_EDITOR = 200
PRIORITY_READER = 100


class Extension(object):

    """Base class of trainer extensions.

    Extension of :class:`Trainer` is a callable object that takes the trainer
    as the argument. It can return a `result dictionary` to inform the result
    with successive extensions. The result is added to
    ``trainer.result[name]``, where ``name`` is the registered name of the
    extension. Result dictionaries should have string keys and numeric values.

    Each extension can be registered to a trainer by the :meth:`Trainer.extend`
    method. This method also takes some configurations of the extension, for
    most of which the extension object itself can provide default values.

    In particular, an extension is combined with a `priority` value. Extension
    has a default value of its priority, while users can manually specify it at
    the registration. There are three standard priorities:

    - ``chainer.trainer.PRIORITY_WRITER``: the priority for extensions that
      return a result dictionary.
    - ``chainer.trainer.PRIORITY_EDITOR``: the priority for extensions that
      edit the result dictionaries.
    - ``chainer.trainer.PRIORITY_READER``: the priority for extensions that
      do not add nor edit the result dictionaries.

    Extensions may support serialization, in which case the trainer includes
    them in the serialization.

    Attributes:
        trigger: Default value of trigger for this extension. It is set to
            ``1, 'iteration'`` by default.
        priority: Default priority of the extension.
        invoke_before_training: Default flag of invoking this extension before
            training starts. It is False by default.

    """
    trigger = 1, 'iteration'
    priority = PRIORITY_READER
    invoke_before_training = False

    @property
    def name(self):
        """Default name of the extension.

        It is the name of the class by default. Implementation can override
        this property, or provide a class attribute to hide it.

        """
        return type(self).__name__

    def __call__(self, trainer):
        """Invoke the extension.

        Implementations should override this operator. This method is called at
        iterations which the corresponding trigger accepts.

        Args:
            trainer (Trainer): Trainer object that calls this operator.

        Returns:
            None or dict: Implementations of this operator can return a result
            dictionary, which has string keys and numeric values.

        """
        raise NotImplementedError

    def serialize(self, serializer):
        pass


def make_extension(trigger=None, name=None, priority=None,
                   invoke_before_training=False):
    """Decorator to convert given functions to extensions.

    This decorator just adds some attributes to a given function. The value of
    the attriubtes are given by the arguments of this decorator.

    Args:
        trigger: Default trigger of the extension. See :class:`Extension` for
            the default value.
        name: Default name of the extension. Unlike the :class:`Extension`
            class, there is no default value for this attribute if this
            argument is omitted.
        priority (int): Default priority of the extension. See
            :class:`Extension` for the default value.
        invoke_before_training (bool): Default flag of invocation before
            training. See :class:`Extension` for the default value.

    Returns:
        Decorator that inserts apprioriate attributes to given functions.

    """
    def decorator(f):
        f.trigger = trigger or Extension.trigger
        if name is not None:
            f.name = name
        f.priority = priority or Extension.priority
        f.invoke_before_training = invoke_before_training
        return f
    return decorator
