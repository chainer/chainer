PRIORITY_WRITER = 300
PRIORITY_EDITOR = 200
PRIORITY_READER = 100


class Extension(object):

    """Base class of all trainer extensions.

    TODO(beam2d): document it.

    """
    trigger = 1, 'iteration'
    priority = PRIORITY_READER
    invoke_before_training = False

    @property
    def name(self):
        return type(self).__name__

    def __call__(self, **kwargs):
        raise NotImplementedError

    def serialize(self, serializer):
        pass


def make_extension(trigger=None, name=None, priority=None,
                   invoke_before_training=False):
    def decorator(f):
        f.trigger = trigger or Extension.trigger
        if name is not None:
            f.name = name
        f.priority = priority or Extension.priority
        f.invoke_before_training = invoke_before_training
        return f
    return decorator
