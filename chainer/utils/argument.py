def check_unexpected_kwargs(kwargs, **unexpected):
    for key, message in unexpected.items():
        if key in kwargs:
            raise ValueError(message)


def parse_kwargs(kwargs, *name_and_values):
    values = [kwargs.pop(name, default_value)
              for name, default_value in name_and_values]
    if kwargs:
        args = ', '.join(["'%s'" % arg for arg in kwargs.keys()])
        raise TypeError('got unexpected keyword argument(s) %s' % args)
    return tuple(values)


def assert_kwargs_empty(kwargs):
    # It only checks if kwargs is empty.
    parse_kwargs(kwargs)
