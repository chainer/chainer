PRIORITY_WRITER = 10000
PRIORITY_EDITOR = 100
PRIORITY_READER = 1


class Extension(object):

    """Base class of all trainer extensions.

    TODO(beam2d): document it.

    """
    default_trigger = 1, 'iteration'
    priority = PRIORITY_READER
    invoke_before_training = False

    @property
    def default_name(self):
        return type(self).__name__

    def __call__(self, **kwargs):
        raise NotImplementedError

    def serialize(self, serializer):
        pass


def make_extension(default_trigger=None, default_name=None, priority=None):
    def decorator(f):
        f.default_trigger = default_trigger or Extension.default_trigger
        f.default_name = default_name
        f.priority = priority or Extension.priority
        return f
    return decorator
