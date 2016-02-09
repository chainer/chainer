class Extension(object):

    """Base class of all trainer extensions.

    TODO(beam2d): document it.

    """
    default_trigger = 1, 'iteration'
    result_action = 'read'
    invoke_before_training = False

    @property
    def default_name(self):
        return type(self).__name__

    def __call__(self, **kwargs):
        raise NotImplementedError

    def serialize(self, serializer):
        pass


def make_extension(default_trigger=None, default_name=None,
                   result_action=None):
    def decorator(f):
        f.default_trigger = default_trigger or Extension.default_trigger
        f.default_name = default_name
        f.result_action = result_action or Extension.result_action
        return f
    return decorator
