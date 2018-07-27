import inspect


def check_unexpected_kwargs(kwargs, **unexpected):
    for key, message in unexpected.items():
        if key in kwargs:
            raise ValueError(message)


def parse_kwargs(kwargs, *name_and_values, **unexpected):
    values = [kwargs.pop(name, default_value)
              for name, default_value in name_and_values]
    if kwargs:
        check_unexpected_kwargs(kwargs, **unexpected)
        caller = inspect.stack()[1]
        args = ', '.join(repr(arg) for arg in sorted(kwargs.keys()))
        message = caller[3] + \
            '() got unexpected keyword argument(s) {}'.format(args)
        raise TypeError(message)
    return tuple(values)


def assert_kwargs_empty(kwargs):
    # It only checks if kwargs is empty.
    parse_kwargs(kwargs)
