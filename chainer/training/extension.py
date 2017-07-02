from chainer.utils import argument


PRIORITY_WRITER = 300
PRIORITY_EDITOR = 200
PRIORITY_READER = 100


class Extension(object):

    """Base class of trainer extensions.

    Extension of :class:`Trainer` is a callable object that takes the trainer
    object as the argument. It also provides some default configurations as its
    attributes, e.g. the default trigger and the default priority. This class
    provides a set of typical default values for these attributes.

    There are three ways to define users' own extensions: inheriting this
    class, decorating closures by :func:`make_extension`, or using any callable
    including lambda functions as extensions. Decorator can slightly reduce the
    overhead and is much easier to use, while this class provides more
    flexibility (for example, it can have methods to configure the behavior).
    Using a lambda function allows one-line coding for simple purposes, but
    users have to specify the configurations as arguments to
    :meth:`Trainer.extend`. For a callable not inheriting this class, the
    default configurations of this class are used unless the user explicitly
    specifies them in :meth:`Trainer.extend` method.

    Attributes:
        trigger: Default value of trigger for this extension. It is set to
            ``(1, 'iteration')`` by default.
        priority: Default priority of the extension. It is set to
            ``PRIORITY_READER`` by default.

    """
    trigger = 1, 'iteration'
    priority = PRIORITY_READER

    @property
    def default_name(self):
        """Default name of the extension.

        It is the name of the class by default. Implementation can override
        this property, or provide a class attribute to hide it.

        """
        return type(self).__name__

    def __call__(self, trainer):
        """Invokes the extension.

        Implementations should override this operator. This method is called
        at iterations which the corresponding trigger accepts.

        Args:
            trainer (Trainer): Trainer object that calls this operator.

        """
        pass

    def __getattr__(self, name):
        if name == 'invoke_before_training':
            raise AttributeError(
                'invoke_before_training has been removed since Chainer '
                'v2.0.0. Use Extension.initialize instead.')
        raise AttributeError('{} object has no attribute {}'.format(
            type(self).__name__, name))

    def finalize(self):
        """Finalizes the extension.

        This method is called at the end of the training loop.

        """
        pass

    def initialize(self, trainer):
        """Initializes up the trainer state.

        This method is called before entering the training loop. An extension
        that modifies the state of :class:`~chainer.training.Trainer` can
        override this method to initialize it.

        When the trainer has been restored from a snapshot, this method has to
        recover an appropriate part of the state of the trainer.

        For example, :class:`~chainer.training.extensions.ExponentialShift`
        extension changes the optimizer's hyperparameter at each invocation.
        Note that the hyperparameter is not saved to the snapshot; it is the
        responsibility of the extension to recover the hyperparameter.
        The :class:`~chainer.training.extensions.ExponentialShift` extension
        recovers it in its ``initialize`` method if it has been loaded from a
        snapshot, or just setting the initial value otherwise.

        Args:
            trainer (Trainer): Trainer object that runs the training loop.

        """
        pass

    def serialize(self, serializer):
        """Serializes the extension state.

        It is called when a trainer that owns this extension is serialized. It
        serializes nothing by default.

        """
        pass


def make_extension(trigger=None, default_name=None, priority=None,
                   finalizer=None, initializer=None, **kwargs):
    """Decorator to make given functions into trainer extensions.

    This decorator just adds some attributes to a given function. The value of
    the attributes are given by the arguments of this decorator.

    See :class:`Extension` for details of trainer extensions. Most of the
    default values of arguments also follow those for this class.

    Args:
        trigger: Default trigger of the extension.
        default_name: Default name of the extension. The name of a given
            function is used by default.
        priority (int): Default priority of the extension.
        finalizer: Finalizer function of this extension. It is
            called at the end of the training loop.
        initializer: Initializer function of this extension. It is called at
            the beginning of the training loop.

    """
    msg = ('invoke_before_training has been removed since Chainer v2.0.0. '
           'Use initializer= instead.')
    argument.check_unexpected_kwargs(kwargs, invoke_before_training=msg)
    argument.assert_kwargs_empty(kwargs)

    if trigger is None:
        trigger = Extension.trigger
    if priority is None:
        priority = Extension.priority

    def decorator(ext):
        ext.trigger = trigger
        ext.default_name = default_name or ext.__name__
        ext.priority = priority
        ext.finalize = finalizer
        ext.initialize = initializer
        return ext

    return decorator
